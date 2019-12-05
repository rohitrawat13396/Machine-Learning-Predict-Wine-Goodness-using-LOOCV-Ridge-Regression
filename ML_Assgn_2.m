X = csvread('trainData.csv',0,1);
y = csvread('trainLabels.csv',0,1);
X_t = csvread('testData_new.csv',0,1);
X_v = csvread('valData.csv',0,1);
y_v = csvread('valLabels.csv',0,1);

X = X';
X_v = X_v';
X_t = X_t';
lambda = [0.01, 0.1, 1, 10, 100, 1000];    
size_X = size(X);
size_X_v = size(X_v);
size_X_t = size(X_t);
size_lambda= size(lambda);
RMSE_train = zeros(1,size_lambda(2),1);
RMSE_val = zeros(1,size_lambda(2));
RMSE_loocv_val = zeros(1,size_lambda(2));
valErrs = zeros(1,size_X_v(2));
cvErrst = zeros(1,size_X(2));
w= zeros(size_X(1),size_lambda(2));
b= zeros(1,size_lambda(2));
obj= zeros(1,size_lambda(2));

disp ('Answers 3.2.1: ');
fprintf('\n\n');
for j= 1:size_lambda(2)
    [w(:,j), b(j), obj(j), cvErrs]= ridgeReg(X,y,lambda(j)) ;
    
    for i = 1:size_X(2)    
               cvErrst(i)=((w(:,j)')*X(:,i) + b(j) - y(i,:));
    end
    for i = 1:size_X_v(2)    
                valErrs(i)=((w(:,j)')*X_v(:,i) + b(j) - y_v(i,:));
    end
    
    [w_v, b_v, obj_v, loocvErrs_v]= ridgeReg(X_v,y_v,lambda(j)) ;     
                
    RMSE_train(j)= sqrt(mean((cvErrst).^2));         %125.2556
    RMSE_val(j)= sqrt(mean((valErrs).^2));         
    RMSE_loocv_val(j)= sqrt(mean((loocvErrs_v).^2)); 
    fprintf('\n\nFor lambda = %f:', lambda(j));
    fprintf('\nRMSE for training data: %f ',RMSE_train(j));
    fprintf('\nRMSE for Validation data: %f ',RMSE_val(j));
    fprintf('\nRMSE for LOOCV Validation data: %f \n',RMSE_loocv_val(j));
end
%plot(lambda,RMSE_train,lambda,RMSE_val,lambda,RMSE_loocv_val);
plot(log10(lambda),RMSE_train,log10(lambda),RMSE_val,log10(lambda),RMSE_loocv_val);
xlabel('Log of Lambda values');
ylabel('RMSE values');
legend('RMSE_train','RMSE_val','RMSE_loocv_val');




best= find(RMSE_loocv_val==min(RMSE_loocv_val));
disp ('Answers 3.2.2: ');
fprintf('\n\n');
fprintf('lambda = %f achieved the best LOOCV performance.',lambda(best));
fprintf('\n');
fprintf('obj value = %f', obj(best));
fprintf('\n');
reg_term= (lambda(best)*((w(:,best)')*w(:,best)));
fprintf('Sum of Square errors = %f', obj(best)-reg_term);
fprintf('\n');
fprintf('Regularization term = %f', reg_term);
fprintf('\n\n');
%disp ('Answers 3.2.3: ');
fprintf('\n\n');
disp ('Answers 3.2.4: ');
fprintf('\n\n');
y_t= zeros(size_X_t(2),2);
%y_t(1,:)= ["Id" "Expected"]; 
for i = 2:size_X_t(2)+1 
    y_t(i,1)= i-2;
    y_t(i,2)= (w(:,best)')*X_t(:,i-1) + b(best);
end    
writematrix(y_t,'predTestLabels.csv');
fprintf('\npredTestLabels.csv is created!\n');



function [w, b, obj, cvErrs] = ridgeReg(X, y, lambda)
    size_X = size(X);
    I_b = eye(size_X(1)+1);
    I_b(size_X(1)+1,size_X(1)+1)= 0;
    one_n= ones(size_X(2),1);
    X_b = [X;one_n'];
    C= X_b*(X_b)' + lambda*I_b;
    d= X_b*y;
    w_b= mldivide(C,d);
    C_inv = inv(C);
    %w_b= C\d;
    w= w_b(1:end-1,1);
    b= w_b(size_X(1)+1,1);
    obj = lambda*(w'*w) +((w_b')*X_b - y')*((X_b')*w_b - y);
    %fprintf('obj: %f ',obj);
    cvErrs= zeros(size_X(2),1);

  for i =  1:size_X(2)  
    x = X_b(:,i);       
    y_i = y(i,:);
    cvErrs(i)= ((w_b')*x-(y_i))/((1-(x')*C_inv*x));
  end
  
end   
 

    

    
    
    