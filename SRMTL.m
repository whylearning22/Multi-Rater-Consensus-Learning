function results = SRMTL(X,Y,R,task_num)

rng(1)
in=numel(X)-1; % internal fold
out=numel(X); % external fold

idx_ext = 1:out;
idx_int = 1:in;


%hyps laughter
%lambda=[10^-7 10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 1 10];
%lambda2=[10^-7 10^-6 10^-5 10^-4 10^-3 10^-2];


%hyps EMOPAIN
%lambda=[10^-4 2*10^-4 3*10^-4 4*10^-4 5*10^-4 6*10^-4 7*10^-4 8*10^-4 9*10^-4 10^-3];
%lambda2=[10^-8 5*10^-8 10^-7 5*10^-7 10^-6 5*10^-6 10^-5 5*10^-5 10^-4 5*10^-4 10^-3 5*10^-3 10^-2 5*10^-2 10^-1 5*10^-1 1 5];



%hyps tuning
lambda=[10^-7 10^-6 10^-5 10^-4 2*10^-4 3*10^-4 4*10^-4 5*10^-4 6*10^-4 7*10^-4 8*10^-4 9*10^-4 10^-3 2*10^-3 3*10^-3 4*10^-3 5*10^-3 6*10^-3 7*10^-3 8*10^-3 9*10^-3 10^-2];
lambda2=[10^-8 5*10^-8 10^-7 5*10^-7 10^-6 5*10^-6 10^-5 5*10^-5 10^-4 5*10^-4 10^-3 5*10^-3 10^-2 5*10^-2 10^-1 5*10^-1 1 5];


posterior_tot=[];
yptot=[];
featImp=zeros(size(X{1,1},2),task_num);
for i=1:out
    
    disp(i)
    
    id_train_ext=find(idx_ext~=i);
    id_test_ext=find(idx_ext==i);
    BagTrain=[];
    labtrain=[];
    for inin=1:numel(id_train_ext)
        BagTrain{inin,1}=X{id_train_ext(inin),1};
        for f=1:task_num
            labtrain{f}{inin}=Y{f}{1,id_train_ext(inin)};
        end
    end
    
    train_ext=cell2mat(BagTrain);
    [train_ext_norm, mu_ext, sigma_ext]=zscore(train_ext);
    sigma_ext(sigma_ext==0)=eps;
    
    test_ext=X{id_test_ext,1};
    
    C = bsxfun(@minus, test_ext, mu_ext);
    test_ext_norm = bsxfun(@rdivide, C, sigma_ext);
    test_ext_norm(isnan(test_ext_norm))=0;
    for f=1:task_num
        labtest_ext{f}=Y{f}{1,id_test_ext};
    end
    for f=1:task_num
        labtrain_ext{f}=cell2mat(labtrain{f}');
    end
    for f=1:task_num
        train_ext_norm_tot{f}=train_ext_norm;
    end
    for f=1:task_num
        test_ext_norm_tot{f}=test_ext_norm;
    end
%% Algorithm 1 row 7    
    for h=1:in
     
        id_train_int=find(idx_int~=h);
        id_test_int=find(idx_int==h);
        BagTrainVal=[];
        labtrainVal=[];
        for inin=1:numel(id_train_int)
            BagTrainVal{inin,1}=BagTrain{id_train_int(inin),1};
            for f=1:task_num
                labtrainVal{f}{inin}=labtrain{f}{1,id_train_int(inin)};
            end
        end
        
        train_int=cell2mat(BagTrainVal);
        [train_int_norm, mu_int, sigma_int]=zscore(train_int);
        sigma_int(sigma_int==0)=eps;
        
        test_int=BagTrain{id_test_int,1};
        C = bsxfun(@minus, test_int, mu_int);
        test_int_norm = bsxfun(@rdivide, C, sigma_int);
        test_int_norm(isnan(test_int_norm))=0;
        for f=1:task_num
            labtest_int{f}=labtrain{f}{1,id_test_int};
        end
        for f=1:task_num
            labtrain_int{f}=cell2mat(labtrainVal{f}');
        end
        for f=1:task_num
            train_int_norm_tot{f}=train_int_norm;
        end
        for f=1:task_num
            test_int_norm_tot{f}=test_int_norm;
        end
        for j=1:length(lambda)
            for z=1:length(lambda2)


                [W_pred_val, C_pred_val]= Logistic_SRMTL(train_int_norm_tot, labtrain_int, R, lambda(j), 0, lambda2(z));
                yptottot=[];
                labtotot=[];
                for tt=1:size(C_pred_val,2)
                    yp_optval{tt}=sign(test_int_norm_tot{tt} * W_pred_val(:,tt) + C_pred_val(tt));
                    yptottot=[yptottot (yp_optval{tt})];
                    labtotot=[labtotot (labtest_int{tt})];
               end
                for pi=1:size(yptottot,1)
                    uu=~isnan(labtotot(pi,:)); 
                    
                    if sum(yptottot(pi,:)==1)> sum(yptottot(pi,:)==-1)
                        mvyp(pi)=1;
                    else
                        mvyp(pi)=-1;
                    end
                    if sum(labtotot(pi,uu)==1)> sum(labtotot(pi,uu)==-1)
                    mvlab(pi)=1;
                    else
                     mvlab(pi)=-1; 
                    end
                     
                end
                CCvaldummy{h}{j,z}=confusionmat(mvlab,mvyp,'order',[-1 1]);
                 end
        end
        
    end   
%% Algorithm 1 row 8    

    for j=1:length(lambda)
        for z=1:length(lambda2)

                CCvaltot=zeros(2,2);
                for h=1:in
                    CCvaltot=CCvaltot+CCvaldummy{h}{j,z};
                end
                [acc_svm_lin_lasso_opt(j,z),macro_svm_lin_lasso_opt(j,z)]=my_micro_macro(CCvaltot);
            
        end
    end

    [v,l]=max(macro_svm_lin_lasso_opt(:));
    [opt_lambda, opt_lambda2]=ind2sub(size(macro_svm_lin_lasso_opt),l);
  
    val_lambda(i)=opt_lambda;
    val_lambda2(i)=opt_lambda2;
%% Algorithm 1 row 11    
    
    [W_pred, C_pred]= Logistic_SRMTL(train_ext_norm_tot, labtrain_ext, R, lambda(opt_lambda), 0, lambda2(opt_lambda2));
    yptottotale=[];
    labtototale=[];
%% Algorithm 1 row 13        
    for tt=1:size(C_pred,2)
        predeach=test_ext_norm_tot{tt} * W_pred(:,tt) + C_pred(tt);
        if sum(predeach)==0 yp_opt{tt}=sign(predeach)-1;
            
        else
            yp_opt{tt}=sign(predeach);
        end
        yptottotale=[yptottotale (yp_opt{tt})];
        labtototale=[labtototale (labtest_ext{tt})];
        if sum(isnan(labtest_ext{tt}))==length(labtest_ext{tt})
            CCtest{tt}{i}=zeros(2,2);
        else
            CCtest{tt}{i}=confusionmat(labtest_ext{tt},yp_opt{tt},'order',[-1 1]);
        end
    end
    mvyptot=[];
    mvlabtot=[];
    for pi=1:size(yptottotale,1)
        uu=~isnan(labtototale(pi,:));
        
        if sum(yptottotale(pi,:)==1)> sum(yptottotale(pi,:)==-1)
            mvyptot(pi)=1;
        else
            mvyptot(pi)=-1;
        end
        if sum(labtototale(pi,uu)==1)> sum(labtototale(pi,uu)==-1)
            mvlabtot(pi)=1;
        else
            mvlabtot(pi)=-1;
        end
    end
    CCtottest{i}=confusionmat(mvlabtot,mvyptot,'order',[-1 1]);
        featImp=featImp+abs(W_pred);
    featImpeach{i,1}=abs(W_pred);
end

CCtotavg=zeros(2,2);
for i=1:out
    CCtotavg=CCtotavg+CCtottest{i};
    
end
[accuracytestavg, macro_mtl_extavg, precision_mtl_extavg, recall_mtl_extavg]=my_micro_macro(CCtotavg);


for tt=1:size(C_pred,2)
    dummy=zeros(2,2);
    for i=1:out
        dummy=dummy+CCtest{tt}{i};
    end
    CCtestot{tt}=dummy;
    
    [accuracytest(tt), macro_mtl_ext(tt), precision_mtl_ext(tt), recall_mtl_ext(tt)]=my_micro_macro(CCtestot{tt});
end
results.featImpeach=featImpeach;          
results.featImptot=featImp;
results.Conftotavg=CCtotavg;
results.accTestavg=accuracytestavg;
results.macroTestavg=macro_mtl_extavg;
results.precisionTestavg=precision_mtl_extavg;
results.recallTestavg=recall_mtl_extavg;


results.Conftot=CCtestot;
results.index_opt_1=val_lambda;
results.index_opt_2=val_lambda2;

results.accTest=accuracytest;
results.macroTest=macro_mtl_ext;
results.precisionTest=precision_mtl_ext;
results.recallTest=recall_mtl_ext;
results.Confsub=CCtottest;
