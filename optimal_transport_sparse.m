function [T,Yhat]=optimal_transport_sparse(X,Y,threshold,lambda,L)
%X is a tuple of locations and densities
%Y is a tuple of locations and densities
%Yhat is a transported Y such that KL(X||Yhat) is minimized
%T is the transportation matrix

nsink=1000;

a = X(:)./sum(X(:));
b = Y(:)./sum(Y(:));

sz=size(X);
dim=size(sz, 2);

coor = [];
if dim==2
    
    [x,y]=meshgrid((1:sz(1)),(1:sz(2)));

    x=permute(x,[2 1]);
    y=permute(y,[2 1]);
    coor = [x(:) y(:)];
    
elseif dim==3
    
    [x,y,z]=meshgrid((1:sz(1)),(1:sz(2)),(1:sz(3)));
    x=permute(x,[2 1 3]);
    y=permute(y,[2 1 3]);
    z=permute(z,[2 1 3]);
    coor = [x(:) y(:) z(:)];
    
else
    
    error('dimension must be 2 or 3');
    
end

coor_x=coor(X(:)>=threshold,:);
coor_y=coor(Y(:)>=threshold,:);
a=a(X(:)>=threshold,:);
b=b(Y(:)>=threshold,:);


M=pdist2(coor_x,coor_y,'euclidean');
M(M>L)=Inf; %Prevent large graph jumps

K=exp(-lambda*M);

U = K.*M;

[D,L,u,v]=sinkhornTransport(a,b,K,U,lambda);
Tsmall= bsxfun(@times,v',(bsxfun(@times,u,K)));

bhat=zeros(size(X(:)));
bhat(X(:)>=threshold) = Tsmall*b;

T=sparse(length(X(:)),length(Y(:)));
T(X(:)>=threshold,Y(:)>=threshold)=Tsmall;


Yhat=reshape(bhat,size(Y));
end