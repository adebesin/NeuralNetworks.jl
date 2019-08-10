import Flux: Flux, LSTM, Chain, normalise
import DataFrames: DataFrames, by, DataFrame, first
import CSV: CSV, File
import Dates: Dates, Date, DateFormat
import MLDataUtils: MLDataUtils, splitobs
# TODO use LSTM neural network to forecast
#=

1) Split train test. Split at January 1, 2010 where train is between January 2, 1980 and December 31, 2009,
2) Normalize the dataset. You only need to fit and transform your training data and just transform your test data

=#

raw = File("resource/usd_inr.csv") |> DataFrame

first(raw, 10)

dateformat = DateFormat("u d, yyyy")
todate = x -> Date(x, dateformat)

raw.Date = todate.(raw.Date)
raw.Price = normalise(raw.Price) # TODO also test scaling between 0 and 1
sort!(raw, cols = [:Date]) # Sort by date ascending

xtrain, xtest = splitobs(raw, at = 0.6)
