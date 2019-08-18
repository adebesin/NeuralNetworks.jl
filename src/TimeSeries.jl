import Flux: Flux, LSTM, Chain, normalise, softmax, Dense, chunk, batchseq, throttle, ADAM, reset!, onehot, onehotbatch, crossentropy, mse, @epochs
import DataFrames: DataFrames, by, DataFrame, first
import CSV: CSV, File
import Dates: Dates, Date, DateFormat
import MLDataUtils: MLDataUtils, splitobs
import Base.Iterators: partition
import LinearAlgebra: transpose
# TODO use LSTM neural network to forecast
#=

1) Split train test. Split at January 1, 2010 where train is between January 2, 1980 and December 31, 2009,
2) Normalize the dataset. You only need to fit and transform your training data and just transform your test data

Notes

Use reset! for sequences?

=#

const epochs = 10

raw = File("../resource/usd_inr.csv") |> DataFrame

dateformat = DateFormat("u d, yyyy")
todate = x -> Date(x, dateformat)

XTrain = Vector{Array{Float64, 1}}
YTrain = Vector{Array{Float64, 0}}

raw.Date = todate.(raw.Date)
raw.Price = normalise(raw.Price) # TODO also test scaling between 0 and 1
sort!(raw, :Date) # Sort by date ascending
prices = convert(Array{Float64, 1}, raw.Price)

function batchprices(prices::Array{Float64, 1}, xlen::Int64)
    xtrain = chunk(prices, xlen)
    ytrain = pop!.(xtrain)
    xtrain, ytrain
end

function batchprices(prices::Array{Int64, 1}, xlen::Int64)
    xtrain = chunk(prices, xlen)
    ytrain = pop!.(xtrain)
    xtrain, ytrain
end

function batchprices(prices::UnitRange{Int64}, xlen::Int64)
    xtrain = chunk(prices, xlen)
    ytrain = pop!.(xtrain)
    xtrain, ytrain
end

xtrain = partition(batchseq(chunk(prices, 5), 0), 50) |> collect
ytrain = partition(batchseq(chunk(prices[2:end], 5), 0), 50) |> collect

const model = Chain(
    LSTM(5, 1),
    Dense(1, 1)
)

function loss(xs, ys)
  l = sum(mse.(model.(xs), ys))
  Flux.truncate!(model)
  return l
end


W = Flux.params(model)
optimisation = ADAM(0.01)

i = 0
@epochs epochs begin
   global  i += 1
    println(i)
    Flux.train!(loss, W, zip(xtrain, ytrain), optimisation)
end
