import Flux: Flux, LSTM, Chain, normalise, softmax, Dense, onehot, chunk, batchseq, throttle, ADAM, reset!, onehot, onehotbatch, crossentropy
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

first(raw, 10)

dateformat = DateFormat("u d, yyyy")
todate = x -> Date(x, dateformat)

raw.Date = todate.(raw.Date)
raw.Price = normalise(raw.Price) # TODO also test scaling between 0 and 1
sort!(raw, cols = [:Date]) # Sort by date ascending

xtrain, xtest = splitobs(raw, at = 0.6)
xtrainprice = convert(Array{Float64, 1}, xtrain.Price)
xtrainpricelength = length(xtrainprice)

seq = [rand(10) for i = 1:10]

typeof(seq)
rand(10)

chunked = chunk(xtrainprice, 117)

const model = Chain(
    LSTM(50, 15),
    Dense(15, 5),
    softmax
)

function loss(x, y)
  l = crossentropy(model(x), y)
  Flux.reset!(m)
  return l
end


output = model.(chunked)

crossentropy(output, xtrainprice)


# Working with sequences
typeof(xtrainprice)

ps = Flux.params(model)
opt = ADAM(0.01)
xtrainprice = convert(Array{Float64, 1}, xtrain.Price)

Flux.train!(loss, ps, xtrainprice, opt)
