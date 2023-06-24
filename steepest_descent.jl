using LinearAlgebra

global h = 0.001

function f_de_alfa(jac::Array,a::Float64,x::Array)
    nabla_ak1 = zeros(length(jac))
    nabla_ak = zeros(length(jac))
    for i = 1:length(jac)
        nabla_ak[i] = jac[i](x...) 
    end
    xkp1 = x - a.*nabla_ak
    for i = 1:length(jac)
        nabla_ak1[i] = jac[i](xkp1...)
    end
    return transpose(nabla_ak)*nabla_ak1
end

function nabla_f_de_alfa(jac::Array,a::Float64,x::Array)
        nabla_f_de_alfa = (f_de_alfa(jac,a+h,x) - f_de_alfa(jac,a-h,x))/2h
        return nabla_f_de_alfa
end

function gradiente(jac::Array,epsilon::Float64,x0::Array,alfa0::Float64)
    k = 0 # contador das iterações da função
    k1 = 0 # contador das iterações do alfa

    x = copy(x0) 
    alfa = alfa0
    erro_f = 1.0 #autoexplicativo
     nabla_ak = zeros(length(jac)) #derivada no ponto x
    x_k = zeros(length(x0))
    a_k = 0.0



while erro_f>epsilon && k<1000
    erro_alfa = 1.0
    for i = 1:length(jac)
        nabla_ak[i] = jac[i](x...) 
    end
    println("nabla $nabla_ak")
        k += 1
        while erro_alfa>1.0e-10 && k1<=1000
            k1+=1
            a_k = alfa - f_de_alfa(jac,alfa,x)/nabla_f_de_alfa(jac,alfa,x)
            erro_alfa = nabla_f_de_alfa(jac,alfa,x)
            alfa = a_k             
        end
    x_k = x - a_k*nabla_ak
    erro_f = max(abs.(nabla_ak)...)
    println(erro_f)
    x = copy(x_k)
    end
    return "$x_k e alfa = $a_k"
end

f1(x1::Float64,x2::Float64) = 1 + 4x1 + 2x2
f2(x1::Float64,x2::Float64) = -1 + 2x1 + 2x2
jacobiana = [f1,f2]
eps = 1.0e-8
x0 = [0.0,0.0]

gradiente(jacobiana,eps,x0,0.0)