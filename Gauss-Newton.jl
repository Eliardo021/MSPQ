using LinearAlgebra #pacote de álgebra linear

# Parâmetros importantes do método do gradiente 
precision_fun = 1.0e-4 #precisão da função gauss-newton  

#=Como entrada, o método de gauss newton precisa de um número m de funções residuais, referentes ao número de experimentos e um número n de coeficiente
das quais dependem as funções residuais, resultando em uma matriz jacobiana de dimensão mxn =#

#Este algoritmo precisa de uma função modelo, dos dados experimentais das variáveis independentes e da dependente e das derivada analítica do modelo para cada variável independente


function optim_d(mder::Array,res::Function,x::Array)
    jacobiana = mder[1](x...)
    for i=2:length(mder)
        jacobiana = cat(jacobiana,mder[i](x...);dims=2)
    end
    fator1 = inv(transpose(jacobiana)*(jacobiana))
    fator2 = fator1 * transpose(jacobiana)
    return fator2*res(x...),jacobiana
end

function gaussnewton(mder::Array,res::Function,x0::Array)
    k_função = 0 
    erro = 1.0
    x_k = 0.0
    x_dummy = copy(x0)
    converged = 0.0
    while erro>precision_fun && k_função<1000
        k_função += 1
        x_k = x_dummy .+ optim_d(mder,res,x_dummy)[1]
        erro = max(abs.(optim_d(mder,res,x_dummy)[2])...)
        println("$x_k $erro $k_função")
        x_dummy = copy(x_k)
        converged = k_função<1000
    end
    if converged == true
        return x_k
    else
        println("ERRO, MUITAS ITERAÇÕES")
        return x_k        
    end
end





