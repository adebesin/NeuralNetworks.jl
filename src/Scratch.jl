import Flux.Tracker
import Flux
import Calculus
import GLM

const flux    = Flux
const tracker = Flux.Tracker
const calc    = Calculus
const glm     = GLM
# calc.gradient(x -> f(x[1]), [2]) TODO compute d²f/dx² = 6


f(x) = 3x^2 + 2x + 1

df(x) = tracker.gradient(f, x; nest = true)[1] # df/dx = 6x + 2 (the derivative = 6x + 2)
df(2)
ceil(calc.derivative(f, 2))

d2f(x) = tracker.gradient(df, x; nest = true)[1] ; # d²f/dx² = 6
d2f(3)

###

f(W, b, x) = W * x + b # Linear function
tracker.gradient(f, 2, 3, 4)

####

#=
"Gradient takes a zero-argument function; no arguments are necessary
because the params tell it what to differentiate"... TODO revise differentiation
=#

W = flux.param(2)
b = flux.param(3)

f(x) = W * x + b
grads = tracker.gradient(() -> f(4), flux.params(W, b))
grads[W]
grads[b]

linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y

#=
Simple linear regression, which tries to predict
an output array y from an input x.
=#

W = rand(2, 5)
b = rand(2)

predict(x) = W * x + b

function loss(x, y)
    Y = predict(x)
    sum((y .- Y) .^2)
end


x, y = rand(5), rand(2)
loss(x, 2)


#=
To improve the prediction we can take the gradients of W
and b with respect to the loss and perform
gradient descent
=#

W = rand(2, 5)
b = rand(2)

W = flux.param(W)
b = flux.param(b)
gs = tracker.gradient(() -> loss(x, y), flux.params(W, b))

delta = gs[W]

# Update the parameter and reset the gradient
tracker.update!(W, -0.1delta)

loss(x, y)

###########################################################

image = load(frog)

readdir("/Users/greade01/Data/cifar/train") |> removehidden

"foo" .+ "bar"

size(channelview(image))

typeof(trainx[1])
typeof(channelview(image))

!startswith("foo", "f")

data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, 1000)]
imgs = MNIST.images()
    size(imgs)


# Partition and batchview are the same except partition returns a lazy seq while batchview is eager
batchview([1,2,3,4], 2)
partition([1,2,3,4], 2) |> collect

import Flux: Data.MNIST, Dense, leakyrelu
import Images: load, channelview, ColorTypes, FixedPointNumbers
import MLDatasets: CIFAR10
import FileIO: load
import Base.Iterators: partition
import MLDataUtils: batchview

const testpath = "/Users/greade01/Data/cifar/test"

function loadimages(path::String)
    images::Array{Array{ColorTypes.RGB4{FixedPointNumbers.Normed{UInt8,8}},2}, 1} = []
    labels::Array{String, 1} = []
    ishidden = file -> !startswith(file, ".")
    files = filter(ishidden, readdir(path))
    for file in files
        imagepath = string(path, "/", file)
        image = load(imagepath)
        label = match(r"(?<=_).*(?=\.)", file).match
        push!(labels, label)
        push!(images, image)
    end
    labels, images
end

function minibatch(images; by = 1000)
    [hcat(vec.(image)...) for image in partition(images, by)]
end


labels, images = loadimages(testpath)
batches = minibatch(images)

N = 32

#TODO Convert from RGB to Grey? the below code fails otherwise?

batches.(Dense(28^2, N, leakyrelu))
decoder = Dense(N, 28^2, leakyrelu) |> gpu


model = Chain(
    LSTM(10, 15),
    LSTM(15, 15),
    Dense(15, 10),
    softmax
)

function loss(x, y)
  l = Flux.mse(model(x), y)
  Flux.reset!(m)
  return l
end

ps = Flux.params(model)
opt = ADAM(0.01)
Flux.train!(loss, ps, xtrain.Price, opt)

const seqlen = 50
const nbatch = 50

Xs = collect(partition(chunk(xtrain.Price, nbatch), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))


scanner = LSTM(length(xtrainprice), 20)
encoder = Dense(20, length(xtrainprice))

function model(x)
    state = scanner.(x)[end]
    reset!(scanner)
    softmax(encoder(state))
end

m = Chain(LSTM(10, 15), Dense(15, 5))


scanner = LSTM(length(xtrainprice), 20)
encoder = Dense(20, length(xtrainprice))
batches(xs, p) = [batchseq(b, p) for b in partition(xs, 50)]

chunked = chunk(xtrainprice, 117)


partitioned = partition(xtrainprice, 1) |> collect

typeof(partitioned)

const chain = Chain(
    LSTM(1, 15),
    Dense(15, 5),
    softmax
)

function model(x)
  state = chain.(x)[end]
  reset!(chain)
end

output = model(xtrainprice)
xtrainprice
loss(x, y) = crossentropy(model(x), y)

ps = Flux.params(chain)
opt = ADAM(0.01)

Flux.train!(loss, ps, partitioned, opt)


batches(xs, p) = [batchseq(b, p) for b in partition(xs, 50)]
crossentropy([1,2,3], [4,5,6])


model = Chain(
  LSTM(xtrainpricelength, 4849),
  LSTM(4849, 4849),
  Dense(4849, xtrainpricelength),
  softmax
)

price = convert(Array{Float64, 1}, raw.Price)
pricelength = length(price)


# Use the previous price to predict the next price
xtrain = price[1:2:pricelength]
ytrain = price[2:2:pricelength]
xtrainlength = length(xtrain)
length(xtrain)

dataiterator = Iterators.repeated((xtrain, ytrain), 5)

model = Chain(
  LSTM(xtrainlength, xtrainlength),
  softmax
)

function loss(xs, ys)
  l = crossentropy(model.(xs), ys)
  Flux.truncate!(m)
  return l
end

ps = Flux.params(model)
opt = ADAM(0.01)

Flux.train!(loss, ps, dataiterator, opt)

dataiterator = Iterators.repeated((xtrain, ytrain), 5)

model = Chain(
  LSTM(xtrainlength, xtrainlength),
  softmax
)

function loss(xs, ys)
  l = crossentropy(model.(xs), ys)
  Flux.truncate!(m)
  return l
end

ps = Flux.params(model)
opt = ADAM(0.01)

Flux.train!(loss, ps, dataiterator, opt)



pricelength = length(price)


# Use the previous price to predict the next price
xtrain = price[1:2:pricelength]
ytrain = price[2:2:pricelength]
xtrainlength = length(xtrain)




# dataiterator = Iterators.repeated((xtrain, ytrain), 1)
const epochs = 2
dataiterator = Iterators.repeated((xtrain, xtrain), epochs)


# Todo create labels

const scanner = Chain(
  LSTM(xtrainlength, xtrainlength),
  softmax
)
const encoder = Chain(
    Dense(xtrainlength, xtrainlength)
)

function model(x)
    state = scanner.(x)[end]
    reset!(scanner)
    softmax(encoder(state))
end

#
# function model(xs, ys)
#   l = crossentropy(model.(xs), ys)
#   Flux.truncate!(m)
#   return l
# end

loss(x, y) = crossentropy(model(x), y) # Compare the model output to the actual value

ps = Flux.params(scanner, encoder)
opt = ADAM(0.01)

Flux.train!(loss, ps, dataiterator, opt)


dataiterator = Iterators.repeated((xtrain, ytrain), epochs)
collect(partition(batchseq(chunk([1,2,3,4,5,6,6,7,8,9,10], 5), 0), 5))

dd = batchseq([[1,2,3], [[1,2,3], [4,5,6], [7,8,9]]], 0)
m.()
function batchprices(prices::Array{Float64, 1})::SubArray{Float64,1,Array{Float64,1}}
    batchedprices = MLDataUtils.slidingwindow(index->index+2, prices, 10, stride=1)
    acc = []
    for t in batchedprices
        append!(acc, collect(t))
    end
    return acc
end

convert(Array{Float64, Float64}, batchedprices[1])
for i in batchedprices
    m.(i)
end


# Todo create labels
toxtrain = x -> convert(XTrain, x)
toytrain = x -> convert(YTrain, x)

toxtrain(xtrain)

const scanner = Chain(
  LSTM(9688, 9688),
  softmax
)
const encoder = Chain(
    Dense(9688, 10)
)

function model(x)
    state = scanner.(x)[end]
    reset!(scanner)
    softmax(encoder(state))
end


loss(x, y) = crossentropy(model(x), y) # Compare the model output to the actual value
ps = Flux.params(scanner, encoder)
opt = ADAM(0.01)

Flux.train!(loss, ps, batchedprices, opt)

seq = [rand(10) for i = 1:10]
m = Chain(LSTM(10, 15), Dense(15, 5))
size(seq)
y = batchprices(prices)
m.(y)

m.([[1,2,3]; [1,2,3]])
dummy = 1:30
batchseq([[1,2,3], [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]], 0)
chunk([1,2,3,4,5], 2)
jj = batchseq([[1,2,3], [4,5,6]], 2)

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


function minibatch(x::Array{Float64, 1}, chunksize::Int64, partitionsize::Int64)
    xchunk, ychunk = chunk(x, chunksize), chunk(x[2:end], chunksize)
    xbatch, ybatch = batchseq(xchunk, 0), batchseq(ychunk, 0)
    xtrain = partition(xbatch, partitionsize) |> collect
    ytrain = partition(ybatch, partitionsize) |> collect
    xtrain, ytrain
end

function minibatch(x::Array{Float64, 1}, range::UnitRange{Int64}, chunksize::Int64, partitionsize::Int64, pad::Float64)
    chunked = chunk(x[range], chunksize)
    batched = batchseq(chunked, pad)
    partition(batched, partitionsize)
end

function minibatch(x::Array{Int64, 1}, range::UnitRange{Int64}, chunksize::Int64, partitionsize::Int64, pad::Int64)
    chunked = chunk(x[range], chunksize)
    batched = batchseq(chunked, pad)
    partition(batched, partitionsize)
end
#TODO split data 70/30 then extract xtrain,ytrain from the 70 and xtest,ytest from the 30
#
# xtrain = partition(batchseq(chunk(prices, 5), 0), 50) |> collect
# ytrain = partition(batchseq(chunk(prices[2:end], 5), 0), 50) |> collect


x1 = minibatch(baz, 2:length(baz), 5, 50, 0) |> collect
x2 = minibatch(baz, 1:length(baz), 5, 50, 0) |> collect
getindex.(collectedxtrain[1:10], 2)
