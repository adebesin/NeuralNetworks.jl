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


price = convert(Array{Float64, 1}, raw.Price)
pricelength = length(price)


# Use the previous price to predict the next price
xtrain = price[1:2:pricelength]
ytrain = price[2:2:pricelength]

dataiterator = Iterators.repeated((xtrain, ytrain), 110)

model = Chain(
  LSTM(xtrainpricelength, 4849),
  LSTM(4849, 4849),
  Dense(4849, xtrainpricelength),
  softmax
)

function loss(xs, ys)
  l = crossentropy(model(xs), ys)
  Flux.truncate!(m)
  return l
end

ps = Flux.params(model)
opt = ADAM(0.01)

Flux.train!(loss, ps, dataiterator, opt)
