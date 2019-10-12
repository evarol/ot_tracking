function T=gmm_transport(props_1,means_1,covs_1,props_2,means_2,covs_2,lambda,L)

for i=1:size(props_1,1)
    for j=1:size(props_2,1)
        M(i,j)=W2(means_1(i,:),covs_1(:,:,i),means_2(j,:),covs_2(:,:,j));
    end
end
a=props_1./sum(props_1);
b=props_2./sum(props_2);
% M(M>L)=Inf; %Prevent large graph jumps
K=exp(-lambda*M);

U = K.*M;

[~,~,u,v]=sinkhornTransport(a,b,K,U,lambda);
T= bsxfun(@times,v',(bsxfun(@times,u,K)));


end


function output=W2(mu_1,sigma_1,mu_2,sigma_2)
%Compute W2 distance metrix between two gaussians (Wasserstein)
output = sum((mu_1-mu_2).^2) + trace(sigma_1 + sigma_2 - 2*ssqrt(sigma_1)*ssqrt(pinv(ssqrt(sigma_1))*sigma_2*pinv(ssqrt(sigma_1)))*ssqrt(sigma_1));

end

function output=ssqrt(X)
%compute the symmetric square root of a symmetric matrix -NOT CHOLESKY
[U,S,V]=svd(X);
output = U*sqrt(S)*V';
end