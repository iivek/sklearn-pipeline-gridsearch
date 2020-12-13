import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from itertools import product
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
import warnings


class PipelineSelectionHelper:
    """
    Compares Pipelines with different transform candidates and a set candidate estimators, over grid of parameters.
    Outputs a nice tabulated summary.
    Extended from https://github.com/davidsbatista/machine-learning-notebooks/blob/master/hyperparameter-across-models.ipynb
    """
    def __init__(self, pipeline, parameters):
        """
        :param pipeline: list of dictionaries.
            Defines alternatives for each Pipeline component together with parameters for grid-search.
            Top-level keys are names of the pipeline stages; their values are (nested) lists that specify alternative
            components that make out the Pipelines.
            Elements comprising each particular pipeline will be taken in the order specified by the outermost
            nesting of the list.
            Use dictionaries of maximum level 2 - the function will not attempt to interpret
            deeper dictionaries (is there a usecase for deeper dicts?)
        :param parameters: dict or list of dictionaries
            Dictionary with parameter names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list will be explored. This enables searching over any sequence
            of parameter settings.
        """
        if self.__dictionary_depth(parameters) > 4:
            print(self.__dictionary_depth(parameters))
            raise ValueError("Parameters dictionary depth is larger than 3 - was that intended?")
        if self.__dictionary_depth(pipeline) > 3:
            raise ValueError("Pipeline dictionary depth is larger than 3 - was that intended?")
        pipeline_set = set(self.__generate_key_value_combinations(pipeline, to_tuple=True))
        parameters_set = set(self.__generate_key_value_combinations(parameters, to_tuple=True))
        if not pipeline_set.issubset(set(parameters_set)):
            missing_parameters = [{k: v} for k, v in pipeline_set - parameters_set]
            warnings.warn("Using default parameters for the following pipeline components: %s" % missing_parameters, UserWarning)
        self.pipeline = pipeline
        self.parameters = parameters
        self.grid_searches = {}

    def __dictionary_depth(self, d, level=1):
        """
        :param d: dictionary
        :param level: internal parameter for the recursion, most probably you don't need to touch it
        :return: maximum depth of the dictionary d (i.e. how deeply nested it is)
        """
        if not isinstance(d, dict) or not d:
            return level
        return max(self.__dictionary_depth(d[k], level + 1) for k in d)

    def __generate_key_value_combinations(self, dictionary, to_tuple=False):
        """
        Helper function.

        :param dictionary:
        :param to_tuple: If set to False, returns list of dictionaries. If set to True returns list of tuples.
        :return: all key-value combinations *at the first level*
        """
        combinations = product(*dictionary.values())
        list_of_dicts = [{**dict(zip(dictionary.keys(), c))} for c in combinations]
        if to_tuple:
            return [(k, v) for f in list_of_dicts for k, v in f.items()]
        return list_of_dicts

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False, groups=None, **fit_params):
        #
        # Forming all pipelines of interest
        #
        cartesian = self.__generate_key_value_combinations(self.pipeline)
        # `cartesian[i]` describes the i-th pipeline:
        # - keys are top-level keys of `self.pipeline` (names of the Pipeline steps)
        # - values are its second-level keys (keys which relate the pipeline step to its parameters)
        for p in cartesian:
            pipe_args = list()
            parameter_grid = dict()
            pipeline_key = list()
            step = 1
            print("Building pipeline:")
            for (k, v) in p.items():
                print("(Step ", step, ") ", k, ": ", v, sep="")
                step += 1
                pipeline_key.append(v)
                # Collecting the arguments to pass on to `Pipeline()` and the parameter grid to pass to GridSearchCV
                pipe_args.append((k, self.pipeline[k][v]))
                # Note that step name got prepended to the parameter name - that is the convention GridSearchCV uses
                _parameters = self.parameters[k].get(v)
                if _parameters is None:
                    _parameters = {}    # No parameters provided - defaulting by passing an empty dict
                for (key, value) in _parameters.items():
                    parameter_grid[k + "__" + key] = value

            # Use transformer caching
            cachedir = mkdtemp()  # Directory for the cache
            memory = Memory(cachedir=cachedir, verbose=0)
            gs = GridSearchCV(Pipeline(pipe_args, memory=memory), parameter_grid, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y, groups, **fit_params)
            pipeline_key = tuple(pipeline_key)
            self.grid_searches[pipeline_key] = gs
            rmtree(cachedir)  # Remove the temporary dir

    def score_summary(self, *validators, sort_by=None, show='all', short_names=True):
        """
        Summarize performance metrics.

        :param sort_by: which column of the frame to sort the results by. Defaults to mean of the first specified
            performance metric (of the last processed GridSearchCV object). Recommended: explicitly set the value.
        :param show: 'all' outputs all parameter grid searches. 'pipeline' limits the output to a single best
            paramater setup per pipeline
        :param short_names: whether to prepend name of the corresponding pipeline stage to the parameter name
        :param validators: instances of PipelineValidationHelper, whose output will be merged with `self` to
            produce a unique dataframe.
        :return: frame holding the summary
        """

        def score_statistics(suffix):
            # Putting the descriptors in relation with the actual aggregate functions
            stats = {'min': pd.DataFrame.min,
                     'max': pd.DataFrame.max,
                     'mean': pd.DataFrame.mean,
                     'std': pd.DataFrame.std,
                     }
            return dict((k+'_'+suffix, v) for k, v in stats.items())

        # Merging results of this class and (optional) other instances passed as *arg
        # Because it's not safe to directly merge the two 'grid_search' members (keys may be the same), let's simply
        # loop over all the grid_search variables and collect the numbers.
        # No explicit mechanism against duplicates - they will be included in the output table.
        amalgamated = list()
        amalgamated.append(self.grid_searches)
        amalgamated.extend([v.grid_searches for v in validators])

        summary = []
        parameter_columns = {}  # Storage for columns in summary which hold grid-search parameter values
        for grid_searches in amalgamated:
            for grid_search_key in grid_searches:
                parameters = grid_searches[grid_search_key].cv_results_['params']
                scores = np.empty((0, len(parameters)))  # pd.DataFrame(columns=parameters)

                # Collecting metrics for each (parameter set, CV
                #
                # Note the '_test' below - we are only interested in performance on the test set
                keys = ['split{}_test_{}'.format(fold, score)
                        for (score, fold) in product(grid_searches[grid_search_key].scoring, range(grid_searches[grid_search_key].n_splits_))]
                for key in keys:
                    r = grid_searches[grid_search_key].cv_results_[key]
                    scores = np.append(scores, [r], axis=0)

                scores = np.transpose(scores)
                individual_performances = pd.DataFrame(scores, columns=keys)
                score_keys = {score: ['split{}_test_{}'.format(fold, score)
                                      for fold in range(grid_searches[grid_search_key].n_splits_)]
                              for score in grid_searches[grid_search_key].scoring}
                # scores = list(score_keys.keys()) # Here's how you would get all metrics
                # list(score_statistics(scores[0]).keys()) # Here's how you would get col names of stats for a metric

                # Calculate statistics/aggregates over the columns corresponding to the same CV
                d = {}
                for key in score_keys.keys():
                    subframe = individual_performances[score_keys[key]]
                    dict_of_stats = dict((s, fun(subframe, axis=1)) for s, fun in score_statistics(key).items())
                    d.update(dict_of_stats)

                # Glue the pipeline description, score statistics and pipeline parameters together
                #
                statistics = pd.DataFrame(d)
                parameters = pd.DataFrame(parameters)
                pipeline = pd.DataFrame([{"pipeline": grid_search_key} for k in range(0, statistics.shape[0])])
                summary_inner = pd.concat([pipeline, statistics, parameters], axis=1)
                summary.append(summary_inner)
                # Hacky: storing unique parameter columns by storing them as key, disregarding the actual value
                parameter_columns.update(dict((col,None)for col in list(parameters)))

        summary = pd.concat(summary, axis=0, ignore_index=True, sort=False)
        # At this point, summary is pretty nicely formatted - if you would like to do any changes in the code,
        # this is a good point to start inspecting and hacking from.

        # Handle `short_names` argument
        if short_names:
            # Remove prefixes (which define the related pipeline stage) from parameter names
            parameter_columns_no_prefix = [c.split('__')[-1] for c in parameter_columns]
            summary.rename(index=str, columns=dict(zip(parameter_columns, parameter_columns_no_prefix)), inplace=True)

        # Handle `sort_by` argument
        if sort_by is None:
            # Fallback to default value
            scores = list(score_keys.keys())
            sort_by = list(score_statistics(scores[0]).keys())[2]   # Mean

        # Handle `show` argument
        def show_all(summary):
            return summary

        def show_pipeline(summary):
            # Getting the best performing parameter setup per pipeline
            rows = summary[sort_by] == summary.groupby('pipeline', sort=False)[sort_by].transform(max)
            return summary[rows]

        def show_default():
            raise NotImplementedError("Unsupported `show` parameter.")

        def show_flowcontrol(summary, selection):
            # Wrapping the functions in lambdas for lazy evaluation; otherwise the NotImplementedError is raised.
            return {'all': lambda: show_all(summary),
                    'pipeline': lambda: show_pipeline(summary),
                    }.get(selection, lambda: show_default())

        summary = show_flowcontrol(summary, show)()
        summary.sort_values([sort_by], ascending=False, inplace=True)
        return summary


def main():
    # Code from https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html,
    # modified to work with our wrapper class

    from sklearn.decomposition import PCA, NMF
    from sklearn.datasets import load_digits
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from tabulate import tabulate

    # Data
    digits = load_digits()

    # Pipelines
    pipeline = {
        'reduce_dim': {
            'PCA': PCA(iterated_power=7),
            'NMF': NMF()
        },
        'classify': {
            'LinearSVC': LinearSVC(),
            'RandomForestClassifier': RandomForestClassifier(),
        },
    }
    parameters = {
        'reduce_dim': {
            'PCA': {'n_components': [2, 4, 8]},
            'NMF': {'n_components': [2, 4, 8]}
        },
        'classify': {
            'LinearSVC': {'C': [1, 10, 100]},
            # 'RandomForestClassifier': {},
        },
    }
    benchmark = PipelineSelectionHelper(pipeline, parameters)
    benchmark.fit(digits.data, digits.target,
                  cv=StratifiedKFold(n_splits=5, shuffle=True),
                  scoring=('accuracy', 'f1_macro', 'precision_macro', 'recall_macro'),
                  n_jobs=3)
    print(tabulate(benchmark.score_summary(show='all'), headers='keys'))


if __name__ == '__main__':
    main()