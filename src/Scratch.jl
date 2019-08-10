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
