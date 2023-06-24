#=
Método de Newton_Raphson para equações algébricas com uma incógnita e derivadas analíticas
=#

#para usar a função, defina duas funções contendo a equação objetivo e sua derivada analítica
function Newton_Raphson(f::Function,f_d::Function,epsilon::Float64,x0::Float64,t::Bool)
    k = 0 #contador de iterações
    x = x0 #variável auxiliar
    erro = 1.0 #variável de erro 
    while erro>=epsilon
        k +=1
        x_k = x - f(x)/f_d(x)
        erro = abs(x_k-x)
        if k>1000
            println("Erro:Muitas iterações, tente novamente com outro valor inicial")
            break
        end
        x = x_k
    end
    if k>1000 #check para ver se convergiu
        return nothing
    else
        if t == 1
            return trunc(x,digits=1)
        elseif t == 0
            return x
        end
    end    
end

# método de newton-raphson com 2 ou mais variáveis 
# Para utilizar, forneça a função e a jacobiana calculada analiticamente, com a dimensão correta

function Newton_Raphson_multivariável(f::Function,J::Array,epsilon::Float64,x0::Array)
    k=0
    erro = 10
    x = x0
    x_k = zeros(length(x0))
    #verificar se a dimensão da jacobiana é compatível com a variável inicial
    if length(J) != length(x0)
        return "Dimensões da Jacobiana|vetor inicial inconsistentes"     
    end
    while erro>epsilon && k<=1000
        for i = 1:length(J)
            x_k[i] = x[i] - f(x...)/J[i](x...)
        end
        erro = max(abs.(x-x_k)...) #verificar as duas variáveis
        println("Erro= $erro ; Variável = $x_k")
        x .=x_k
        k+=1
    end
    if k>1000
        println("muitas iterações")
        return
    else
        return trunc.(x,digits = 3)
    end
end

function derivada_numérica(f::Function,x0::Float64,h::Float64)
    dx = (f(x0+h)-f(x0-h))/2h
    return trunc(dx,digits=3)
end