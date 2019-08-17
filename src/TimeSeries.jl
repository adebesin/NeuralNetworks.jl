import Flux: Flux, LSTM, Chain, normalise, softmax, Dense, chunk, batchseq, throttle, ADAM, reset!, onehot, onehotbatch, crossentropy
import DataFrames: DataFrames, by, DataFrame, first
import CSV: CSV, File
import Dates: Dates, Date, DateFormat
import MLDataUtils: MLDataUtils, splitobs
import Base.Iterators: partition
# TODO use LSTM neural network to forecast
#=

1) Split train test. Split at January 1, 2010 where train is between January 2, 1980 and December 31, 2009,
2) Normalize the dataset. You only need to fit and transform your training data and just transform your test data

Notes

Use reset! for sequences?

=#

raw = File("resource/usd_inr.csv") |> DataFrame

dateformat = DateFormat("u d, yyyy")
todate = x -> Date(x, dateformat)

XTrain = Vector{Array{Float64, 1}}
YTrain = Vector{Array{Float64, 0}}

raw.Date = todate.(raw.Date)
raw.Price = normalise(raw.Price) # TODO also test scaling between 0 and 1
sort!(raw, cols = [:Date]) # Sort by date ascending


toxtrain = x -> convert(XTrain, x)
toytrain = x -> convert(YTrain, x)


prices = convert(Array{Float64, 1}, raw.Price)

batchedprices = MLDataUtils.slidingwindow(index->index+2, prices, 10, stride=1)
xtrain = first.(batchedprices) |> toxtrain
ytrain = last.(batchedprices) |> toytrain
