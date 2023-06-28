using LinearAlgebra
include("Exercício Proposto 02.jl")

#parâmetros
precisão_fun = 1.0e-8
nu = 2.0

function find_lambda(jac::Array,x_0::Array)
    grad = jac[1](x_0...)
    for i=2:length(x_0)
        grad = cat(grad,jac[i](x_0...);dims=2)
    end
    t1 = transpose(grad)*grad
    if typeof(t1) != Array
        lambda = convert(Float64,t1)
        t1 = [t1]
    else
        lambda = max(diag(t1,0)...)
    end    
    return lambda, grad, t1
end

function opt_delta(jac::Array,res::Function,x::Array,lambda::Float64)
    fun_lamb = find_lambda(jac,x)
    jacobiana = fun_lamb[2]
    parte1 = fun_lamb[3]
    parte2 = inv(parte1 + lambda*Diagonal(parte1))
    parte3 = parte2 * (transpose(jacobiana)*res(x...))
    #println("$parte1 $parte2 $parte3")
    return parte3,jacobiana
end


function levenberg_marq(jac::Array,res::Function,x_0::Array,f_obj::Function)
    k = 0
    erro = 1.0
    x_dummy = copy(x_0)
    x_k = 0.0
    lambda = find_lambda(jac,x_0)[1]
    while erro>precisão_fun && k<1000
        k+=1
        x_k = x_dummy + opt_delta(jac,res,x_dummy,lambda)[1]
        erro = max(abs.(opt_delta(jac,res,x_dummy,lambda)[2])...)
        if f_obj(x_k...)<f_obj(x_dummy...)
            lambda = lambda/nu
        else
            lambda = lambda*nu
        end
        x_dummy = copy(x_k)
        println("$x_k $(f_obj(x_k...)) $k")
    end
    return x_k
end