import Flux: Flux, LSTM, Chain, normalise, softmax, Dense, chunk, batchseq, throttle, ADAM, reset!, onehot, onehotbatch, crossentropy, mse, @epochs, Tracker
import DataFrames: DataFrames, by, DataFrame, first
import CSV: CSV, File
import Dates: Dates, Date, DateFormat
import MLDataUtils: MLDataUtils, splitobs
import Base.Iterators: partition
import LinearAlgebra: transpose
import StatsPlots: plot
# Todo use LSTM neural network to forecast
#=

1) Split train test. Split at January 1, 2010 where train is between January 2, 1980 and December 31, 2009,
2) Normalize the dataset. You only need to fit and transform your training data and just transform your test data

Notes

Use reset! for sequences?

=#


const epochs = 10
const chunksize = 5

struct Batch
    x::Array{Float64, 1}
    range::UnitRange{Int64}
    chunksize::Int64
    partitionsize::Int64
    pad::Float64
end

function minibatch(batch::Batch)
    chunked = chunk(batch.x[batch.range], batch.chunksize)
    batched = batchseq(chunked, batch.pad)
    partition(batched, batch.partitionsize)
end

raw = File("resource/usd_inr.csv") |> DataFrame

dateformat = DateFormat("u d, yyyy")
todate = x -> Date(x, dateformat)

XTrain = Vector{Array{Float64, 1}}
YTrain = Vector{Array{Float64, 0}}


raw.Date = todate.(raw.Date)
raw.Price = normalise(raw.Price) # TODO also test scaling between 0 and 1
sort!(raw, :Date) # Sort by date ascending
prices = convert(Array{Float64, 1}, raw.Price)

train, validation, test = splitobs(prices, (.5, .3))
plot(train)

xbatch = Batch(train, 1:length(train), chunksize, 50, 0)
ybatch = Batch(train, 2:length(train), chunksize, 50, 0)


xtrain = minibatch(xbatch)
ytrain = minibatch(ybatch)

#TODO softmax?
const model = Chain(
    LSTM(chunksize, 1),
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

collectedxtrain = xtrain |> collect
collectedytrain = ytrain |> collect

trainmodel = model.(train)
validatemodel = model.(validation)

concatenated = vcat(trainmodel, validatemodel)

plot(Tracker.data.(first.(concatenated)))
plot(vcat(train, validation))

trainmodel = model.(validation)
