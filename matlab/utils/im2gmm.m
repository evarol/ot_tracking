function [means,covs,props]=im2gmm(im,numsample,k);

p=im(:)/sum(im(:));

sz=size(im);

if length(size(sz))==3
    [x,y,z]=meshgrid((1:sz(1)),(1:sz(2)),(1:sz(3)));
    x=permute(x,[2 1 3]);
    y=permute(y,[2 1 3]);
    z=permute(z,[2 1 3]);
    coor = [x(:) y(:) z(:)];
elseif length(size(sz))==2
    [x,y]=meshgrid((1:sz(1)),(1:sz(2)));
    x=permute(x,[2 1]);
    y=permute(y,[2 1]);
    coor = [x(:) y(:)];
end

y = randsample((1:length(coor))',numsample,true,p);

sample=coor(y,:);
% I=zeros(size(im));
% for i=1:size(sample,1)
% I(sample(i,2),sample(i,1))=I(sample(i,2),sample(i,1))+1;
% end

GMModel = fitgmdist(sample,k); 
means=GMModel.mu;
covs=GMModel.Sigma;
props=GMModel.ComponentProportion';
end