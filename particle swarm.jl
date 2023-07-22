using Random
using Distributions

rng = RandomDevice()
niter = 10000
w = 0.4
w_max = 1.0
w_min = 0.0
c1 = 2.0
c2 = 2.0
std_range = Uniform(0.0,1.0)
prec = 1e-4

function initial_particles(size::Int64=1,range::Array = [-1.0,1.0],n_particles::Int64=0)
    if n_particles == 0
        n_particles = 20*size
    end
    flock = zeros(n_particles)
    population = map(flock) do t
        rand(rng,Uniform(range[1],range[2]),size)
    end
    velocity = map(flock) do t
        rand(rng,Uniform(range[1],range[2]),size)
    end
    return population,velocity
end

function get_best(population::Array,f_obj::Function,best_p::Array=[0])
    if best_p == [0]
        best_p = copy(population)
    end
    
    fitness = map(best_p) do t
        f_obj(t...)
    end

    globa_best = best_p[argmin(fitness)]

    for i in 1:length(population)
        t = f_obj(population[i]...)
        t_b = f_obj(best_p[i]...)
        if t<t_b
            best_p[1]=population[i]
        end
        if t<f_obj(globa_best...)
            globa_best = population[i]
        end
    end
    return globa_best,best_p
end

function Velocity(velocity::Array,best_p::Array,global_best::Array,population::Array,w::Float64=0.9)
    new_velocity = map(velocity,best_p,population) do x,y,z
        w * x .+ c1*rand(rng,std_range)*(y.-z) + c2*rand(rng,std_range)*(global_best.-z)
    end
    return new_velocity
end

function inertial(k::Int64)
    w_new = w_max - (w_max - w_min)*(k-1)/(niter-1)
    return w_new
end

function get_new_population(population::Array,velocity::Array,bounds::Array)
    new_population = map(population,velocity) do x,y
        x .+ y
    end

    for j in 1:size(new_population)[1]
        for i in 1:size(new_population[j])[1]
            if new_population[j][i]<bounds[1]
                new_population[j][i] = bounds[1]
            elseif new_population[j][i]>bounds[2]
                new_population[j][i] = bounds[2]
            end
        end
    end
    return new_population
end

function particle_swarm(f_obj::Function,range::Array)
    dim = length(methods(f_obj)[1].sig.parameters)-1
    status = initial_particles(dim,range)
    population = copy(status[1])
    velocity = copy(status[2])
    global_best = get_best(population,f_obj)[1]
    best_allpop = copy(population)
    counter = 1
    flag = 0

    w = inertial(counter)
    population_new = zeros(length(population))
    velocity_new = zeros(length(velocity))
    
    
    for i in 1:niter
        velocity_new = Velocity(velocity,best_allpop,global_best,population,w)
        population_new = get_new_population(population,velocity_new,range)
        status=get_best(population_new,f_obj,best_allpop)
        global_best = copy(status[1])
        best_allpop = copy(status[2])
        counter += 1
        w = inertial(counter)
        if f_obj(global_best...)<prec
            break
        end
        velocity = velocity_new
        population = population_new
        #println(global_best)
    end
    flag =  f_obj(global_best...)<prec
    respostas = ["Não Convergiu","Convergiu"]
    println("$(respostas[Int64(flag)+1])")
return global_best, counter-1, f_obj(global_best...)
end