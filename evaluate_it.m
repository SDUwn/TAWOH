function [eva,train_time_round] = evaluate_it(XTrain,YTrain,LTrain,XQuery,LQuery,train_param)
    eva=zeros(1,train_param.nchunks);
    train_time_round=zeros(1,train_param.nchunks);

    %% initialization
    X=cell2mat(XTrain);
    anchor=X(randsample(2000,1000),:);
    HTrain=[];
    
    for chunki=1:train_param.nchunks
%         fprintf('--chunk---%3d\n',chunki);
        
        XTrain_t=XTrain{chunki,:};
        YTrain_t=YTrain{chunki,:};
        XQuery_t=XQuery{chunki,:};
        LQuery_t=LQuery{chunki,:};

        tic 
        if chunki==1
            [DD,BB,W1]=train0(XTrain_t',YTrain_t',train_param,anchor');
        else
            [DD,BB,W1]=train(XTrain_t',YTrain_t',DD,BB,train_param,anchor');
        end
        train_time_round(1,chunki)=toc;
        
%         fprintf('test beginning\n');
        B=cell2mat(BB)';
        HTrain=single(B>0);
        XQuery_t=Kernelize(XQuery_t,anchor);
        
        HTest=single(XQuery_t*W1'>0);
        
        Lbase=cell2mat(LTrain(1:chunki,:));
        Aff = affinity([], [], Lbase, LQuery_t, train_param);
        
        train_param.metric = 'mAP';
        eva(1,chunki)  = evaluate(HTrain, HTest, train_param, Aff);
        
    end
    
end


