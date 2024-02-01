

"""

# interface from https://github.com/igaloly/slice_finder

# Initialize Genetic Algorithm Slice Finder with desired data connector and data structure
slice_finder = GAMuPlusLambdaSliceFinder(
    data_connector=PandasDataConnector(
        df=df,
        X_cols=df.drop(['pred', 'target'], axis=1).columns,
        y_col='target',
        pred_col='pred',
    ),
    data_structure=FlattenedLGBMDataStructure(),
    verbose=True,
    random_state=42,
)
"""


    














