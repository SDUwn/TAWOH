function [DD,BB,F]=train(XTrain_t,YTrain_t,DD,BB,param,anchor)

    %% set the parameters
    nbits = param.current_bits;
    alpha = param.alpha;
    beta = param.beta;
    kesi = param.kesi;
    yita = param.yita;
    sigma=param.sigma;
    phi=param.phi;
    lamda=param.lamda;
    
    theta1=0.5;
    theta2=0.5;

    %% get the dimensions of features
    n= size(XTrain_t,2);
    dX=size(anchor,2);
%     dX = size(XTrain_t,1);
    dY = size(YTrain_t,1);


    %% initialization
    B = sign(randn(nbits, n));
    W1 = randn(dX, nbits);
    W2 = randn(dY, nbits);
    P = randn(dY,nbits);
    L_new=randn(dY,n);

    XTrain_t=Kernelize(XTrain_t',anchor');
    XTrain_t=XTrain_t';
    
    %% iterative optimization
        for iter = 1:param.iter
            % update H_new
            Z=theta1*W1'*XTrain_t+theta2*W2'*YTrain_t...
            +nbits*beta*((B*L_new'+DD{1,1})*L_new)...
            +sigma*P'*L_new...
            +alpha*B;
        
            Temp = Z*Z'-(1/n)*Z*ones(n,1)*ones(1,n)*Z';
            [~,Lmd,OO] = svd(Temp); clear Temp
            idx = (diag(Lmd)>1e-6);
            O = OO(:,idx); 
            O_ = orth(OO(:,~idx));
            N = Z'*O/(sqrt(Lmd(idx,idx)))-(1/n)*ones(n,1)*(ones(1,n)*Z')*O/(sqrt(Lmd(idx,idx)));
            N_ = orth(randn(n,nbits-length(find(idx==1))));
            H_new = sqrt(n)*[O O_]*[N N_]';
            
            % update W
            W1=theta1*(XTrain_t*H_new'+DD{1,2})/(theta1*(H_new*H_new'+DD{1,4})+lamda*eye(nbits)) ;
            W2=theta2*(YTrain_t*H_new'+DD{1,3})/(theta2*(H_new*H_new'+DD{1,4})+lamda*eye(nbits)) ;
            
            % update R
            R=(kesi*(L_new*YTrain_t'+DD{1,5})+yita*(YTrain_t*YTrain_t'+DD{1,6}))/((kesi+yita)*(YTrain_t*YTrain_t'+DD{1,6}));
 
            % update L_new
            L_new=((kesi+sigma)*eye(dY))^(-1)*(kesi*R*YTrain_t+sigma*P*H_new);
            
            %update F
           F=(B*XTrain_t'+DD{1,7})/(XTrain_t*XTrain_t'+DD{1,8}+lamda*eye(dX));
           
           %update P
           P=(L_new*H_new'+DD{1,9})/(H_new*H_new'+DD{1,4}+lamda*eye(nbits));
           
           %update B
           B = sign(alpha*H_new+beta*nbits*(H_new*L_new'+DD{1,10})*L_new+phi*F*XTrain_t);
           
            %update theta
            theta1=(sqrt(sum(sum((XTrain_t-W1*H_new).^2)))) / (sqrt(sum(sum((XTrain_t-W1*H_new).^2)))+sqrt(sum(sum((YTrain_t-W2*H_new).^2))));
            theta2=(sqrt(sum(sum((YTrain_t-W2*H_new).^2)))) / (sqrt(sum(sum((XTrain_t-W1*H_new).^2)))+sqrt(sum(sum((YTrain_t-W2*H_new).^2))));
        
        end
        DD{1,1} = B*L_new';
        DD{1,2} = XTrain_t*H_new';
        DD{1,3} = YTrain_t*H_new';
        DD{1,4}= H_new*H_new';
        DD{1,5} = L_new*YTrain_t';
        DD{1,6} = YTrain_t*YTrain_t';
        DD{1,7} = B*XTrain_t';
        DD{1,8}=XTrain_t*XTrain_t';
        DD{1,9} = L_new*H_new';
        DD{1,10}=H_new*L_new';
        %%
        BB{1,end+1}=B;
end
