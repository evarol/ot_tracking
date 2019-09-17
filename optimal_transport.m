function [T,Yhat]=optimal_transport(X,Y,lambda,L)
%X is a tuple of locations and densities
%Y is a tuple of locations and densities
%Yhat is a transported Y such that KL(X||Yhat) is minimized
%idx is locations of Y in Yhat i.e. the permutation
nsink=1000;

a = X(:)./sum(X(:));
b = Y(:)./sum(Y(:));

sz=size(X);
[x,y]=meshgrid((1:sz(1)),(1:sz(2)));

x=permute(x,[2 1]);
y=permute(y,[2 1]);
coor = [x(:) y(:)];

M=squareform(pdist(coor,'euclidean'));
M(M>L)=Inf; %Prevent large graph jumps

K=exp(-lambda*M);

U = K.*M;

[D,L,u,v]=sinkhornTransport(a,b,K,U,lambda);
T= bsxfun(@times,v',(bsxfun(@times,u,K)));

bhat = T*b;
Yhat=reshape(bhat,size(Y));
end