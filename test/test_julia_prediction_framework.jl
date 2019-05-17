#=
test_julia_prediction_framework:
- Julia version: 1.1.0
- Author: Franz Nowak
- Date: December 2018-May 2019
=#
include("index/julia_prediction_framework.jl")

# Relu function
@test relu([1,2,-3])==[1,2,0]

# binary search, positive example
@test binary_search([1,2,3,4,5],3,1,5)==3

# binary search, negative example
@test_throws KeyError binary_search([1,2,3,4,5],0,1,5)==3

# exponential search, positive example
@test exponential_search([1,2,3,4,5],3,1)==3

# exponential search, negative example
@test_throws KeyError exponential_search([1,2,3,4,5],0,1)==3