
"Composite DataFrame Pattern"
    * Holds 1+ dataframes as privates
    * All methods return dataframes
    * No business logic, just joins and filters of the owned dataframes

"Static Worker Class Pattern" (Deprecated?)
    * operates on a parallelizable unit
    * Uses staticmethods that can be tested independently
    * Weaves the staticmethods together with self. chains to make debugginer serialization easier
        TODO: Explain
    * Are used by a worker free-function in the same module
