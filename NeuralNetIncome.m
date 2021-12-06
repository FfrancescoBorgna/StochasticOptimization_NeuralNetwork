clear all
clc
close all

%carico il set ottenuto dall'altro script
load('OptimizationSet.mat');

N = 2;
H = 3;
P = length(m_val); %lunghezza del data set


%Create all the neurons
node_in = zeros(N,1);
node_in(N) = 1; %bias
node_h = zeros(H,1);
node_h(H) = 1; %bias

%create the weights
w = 0.4*ones(N,H-1);
k = 0.3*ones(H,1);

%output values
out = zeros(P,1);
vph = zeros(P,H);

tol = 0.005;
%Prove%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%singleInForwardProp([1;1],node_h,w,k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Prepare the dataset
%training values
indexes = [1,3,4,5,7,8,9,10];
in_training = [m_val(indexes)',ones(length(indexes),1)]; %aggiungiamo il bias
y_training = M(indexes,2); %M(:,2) è relativo a 10000 simulazioni
%Normalize y_training
y_max = max(M(:,2));
y_min = min(M(:,2));
y_training = (y_training-y_min)/(y_max-y_min);

%testing values
indx_test = [2,6,11];
in_test = [m_val(indx_test)',ones(3,1)]; %aggiungiamo il bias
y_test = M(indx_test,2); %M(:,2) è relativo a 10000 simulazioni

% %training
[w_t,k_t,iterations]=training(in_training,node_h,w,k,tol,y_training) 
out_pred = singleInForwardProp([0.2,1],node_h,w_t,k_t)
[out_predicted,] = forwardProp(in_test,node_h,w_t,k_t)


function [w_trained,k_trained,m] = training(in,node_h,w,k,tol,y)
    P = length(in(:,1));
    H = length(k);
    N = length(in(1,:));
    
    out = zeros(P,1);
    vph = zeros(P,H);
    %SSE
    SSE_old = 10000;
    SSE_new = 1000;
    m = 1;
    %m=2;
    while(norm(SSE_new-SSE_old) > tol)
        
        [out,vph] = forwardProp(in,node_h,w,k);
        mu = 500/(1000+m);
        %mu = log(m+1)/m;
        %Update the w weights
        for i = 1:N
            for h=1:(H-1)
                w(i,h) = w(i,h) + mu *gradW(y,out,k,vph,in,i,h);
            end
        end
        
        %Update the k weights
        for h=1:H
            k(h) = k(h)+mu*gradK(y,out,vph,in,h);
        end
        
        SSE_old = SSE_new;
        SSE_new = sum((y-out).^2);
        m = m+1;
    end
    w_trained = w;
    k_trained = k;
    
end
function [out,vph] = forwardProp(in,node_h,w,k)
    %in è il vettore degli input del data set compreso il bias
    P = length(in(:,1));
    H = length(k);
    out = zeros(P,1);
    vph = zeros(P,H);
    for p = 1:length(in)
        [out(p),vph(p,:)] = singleInForwardProp(in(p,:),node_h,w,k);
    end
end
function [out_p,vp] = singleInForwardProp(in, node_h, w,k)
    %%out_p = l'output relativo al singolo input del data set
    %vp = output dei singoli hidden neurons per l'input p del data set
    %
    %in è il singolo input dal dataset
    for h = 1:(length(node_h)-1)
        node_h(h) = sum(w(:,h).*in');
    end
    sigmoide = @(u) (1+exp(-u)).^(-1);
    vp = arrayfun(sigmoide,node_h);
    out_p = sum(vp.*k);  
end

function [grad] = gradW(y,o,k,v,x,i,h)
%y, vettore con gli output del data set
%o, vettore con gli output della rete
%k, vettore con i pesi dall hidden al output
%v, matrice PxH con gli output degli hidden layer per ogni campione del dataset
%x, matrice PxN contenente tutti gli input del dataset (include il bias)
%i, indice i
%h indice h
    grad = 0;
    for p=1:length(x(:,1)) %1:P
        grad = grad + (y(p)-o(p))*k(h)*v(p,h)*(1-v(p,h))*x(p,i);
    end
    grad = 2*grad;
end
function [grad] = gradK(y,o,v,x,h)
%y, vettore con gli output del data set
%o, vettore con gli output della rete
%k, vettore con i pesi dall hidden al output
%v, matrice PxH con gli output degli hidden layer per ogni campione del dataset
%x, matrice PxN contenente tutti gli input del dataset (include il bias)
%i, indice i
%h indice h
    grad = 0;
    for p=1:length(x(:,1)) %1:P
        grad = grad + (y(p)-o(p))*v(p,h);
    end
    grad = 2*grad;
end
