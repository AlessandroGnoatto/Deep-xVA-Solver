T = xlsread('exposureBasketOption.xlsx');
lambdaC = 0.1;
lambdaB = 0.01;
r = 0.01;
recoveryB = 0.4;
recoveryC = 0.3;

deltaT = 1/100;

for i = 1 :size(T,2)
    EPE(i) = mean(max(T(:,i),0));
    ENE(i) = mean(max(-T(:,i),0));
    CVA(i) = ENE(i) * lambdaB * exp(-(r+lambdaC+lambdaB)*deltaT* i);
    DVA(i) = EPE(i) * lambdaB * exp(-(r+lambdaC+lambdaB)*deltaT* i);
end

disp("DVA short call - vHat > 0")
disp(sum(DVA)*(1-recoveryB)*deltaT)
disp("CVA short call - vHat > 0")
disp(sum(CVA)*(1-recoveryC)*deltaT)



