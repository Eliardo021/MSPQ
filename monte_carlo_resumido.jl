using Random
using Distributions #novamente, é preciso instalar este módulo, ele não vem nativamente com Julia
rng = RandomDevice()
#para monte carlo simples
np = 100000 #numero de pontos aleatórios gerados no range de busca
#para monte carlo complicada


function monte_carlo_simples(s_range::Uniform{Float64},fun::Function,size::Int64,n_samples::Int64=np)
    m_pontos = Float64[]
    f_pontos = Float64[]
    k = zeros(size)
    for i=1:n_samples
        k_i = map(x -> rand(rng,s_range),k)
        m_pontos = vcat(m_pontos,[k_i])
        f_pontos = vcat(f_pontos,fun(k_i...))
    end
    return m_pontos[argmin(f_pontos)],min(f_pontos...)
end

#versão otimizada com multi threading, a depender da cpu do processador pode diminuir muito o tempo de cálculo.
#Para 4 threads, diminuiu em 3 vezes o tempo necessário para calcular em relação à uma única thread
#Como o padrão do REPL é iniciar com apenas 1 thread, é necessário mudar a configuração do visual studio code
#para inicializar com todas as threads disponíveis em settings.json
#colocar o valor "auto" em num_threads ou iniciando o script com a opção '--threads=auto'

function optim_montecarlo(s_range::Array,fun_obj::Function,Nx::Int64)
    s_range_dist = Uniform(s_range[1],s_range[2])
    num_threads = Threads.nthreads()
    samp_chunks = fill(Int64(np/num_threads),num_threads)
    tasks = map(samp_chunks) do samp_chunks
        Threads.@spawn monte_carlo_simples(s_range_dist,fun_obj,Nx,samp_chunks)
    end
    chunk_results = fetch.(tasks)
    x_k = chunk_results[argmin(map(x -> fun_obj(x...),chunk_results))]
    return x_k
end



