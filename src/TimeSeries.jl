import Flux: Flux, LSTM, Chain, normalise, softmax, Dense, onehot, chunk, batchseq, throttle, ADAM
import DataFrames: DataFrames, by, DataFrame, first
import CSV: CSV, File
import Dates: Dates, Date, DateFormat
import MLDataUtils: MLDataUtils, splitobs
import Base.Iterators: partition
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

scanner = LSTM(length(xtrain.Price), 20)
encoder = Dense(20, length(xtrain.Price))

function model(x)
    state = scanner.(x.data)[end]
    reset!(scanner)
    softmax(encoder(state))
end


loss(x, y) = crossentropy(model(x), y)

# Working with sequences
model.(xtrain.Price)

ps = Flux.params(scanner, encoder)
opt = ADAM(0.01)
xtrainprice = convert(Array{Float64, 1}, xtrain.Price)

Flux.train!(loss, ps, xtrainprice, opt)
