using Random
using Distributions

#parâmetros importantes
ntmax = 1000
nxmax = 50
cooling_rate = 0.75 #Taxa de resfriamento
Temperatura_inicial = 400.0
delta = 1.0 #padrão para encontrar pontos vizinhos, pode ser alterado de acordo com a dimensão da variável
sigma = 1.0 #desvio padrão da gaussiana
rng = RandomDevice()

function resfriamento_linear(T,alfa::Float64=cooling_rate)
    Tk1 = alfa*T
    return Tk1
end

function get_new_neighbor_rand(x::Array,d::Float64=delta)
    xk = map(x) do t
        t + rand(rng,Uniform(-d,d))
    end
    return xk 
end

function get_new_neighbor_gauss(x::Array,s::Float64=sigma)
    xk = map(x) do t
        t + rand(rng,Normal(t,s))
    end
    return xk
end

function get_best_neighbor(T::Float64,f_obj::Function,x::Array,d::Float64=delta)
    best_e = f_obj(x...)
    e_x = 0.0
    x_best = copy(x)
    x_k = zeros(length(x))
    for i in 1:nxmax
        x_k = get_new_neighbor_rand(x_best,d)
        e_x = f_obj(x_k...)
        df = e_x - best_e

        if df<0 || rand(rng,Uniform(0.0,1.0)) < exp(-df/T)
            x_best = copy(x_k)
            best_e = e_x
        end
        #println(x_best, best_e)
    end
    return x_best,best_e
end

function sim_annealing(f_obj::Function,range::Array)
    T_i = Temperatura_inicial
    size = length(methods(f_obj)[1].sig.parameters) - 1
    x_i = rand(Uniform(range[1],range[2]),size)
    x_t = copy(x_i)
    status = Array[]
    energia = f_obj(x_t...)

    for i in 1:ntmax
        status = get_best_neighbor(T_i,f_obj,x_t,0.1)
        energia = status[2]
        x_t = status[1]
        T_i = resfriamento_linear(T_i)
        #println(x_t,energia)
    end
    return x_t,energia,T_i
end