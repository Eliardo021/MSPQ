using Random
using Distributions
rng = RandomDevice()
#para monte carlo simples
np = 100000 #numero de pontos aleatórios gerados no range de busca
#para monte carlo complicada
Npoints = 100
Niter = 100 #numero máximo de iterações
precision = 1.0e-5 #precisão



function monte_carlo_simples(s_range::Uniform{Float64},fun::Function,size::Int64)
    m_pontos = Float64[]
    f_pontos = Float64[]
    k = zeros(size)
    for i=1:np
        k_i = map(x -> rand(rng,s_range),k)
        m_pontos = vcat(m_pontos,[k_i])
        f_pontos = vcat(f_pontos,fun(k_i...))
    end
    return m_pontos[argmin(f_pontos)]
end


function optim_montecarlo(s_range::Array,fun_obj::Function,Nx::Int64)
    s_range_k = Uniform(s_range[1],s_range[2])
    iter = 0
    size = 2.0
    x_k = zeros(Nx)
    for iter=1:Niter
        x_k = gerar_pontos(s_range_k,fun_obj,Nx)
        s_range_k = Uniform(-(abs(min(x_k...))+size),max(abs.(x_k)...)+size)
        size = size/2
        println("$s_range_k $size $x_k")
        if abs(fun_obj(x_k...))<precision
            break
        else 
            continue
        end
    end
return x_k
end

