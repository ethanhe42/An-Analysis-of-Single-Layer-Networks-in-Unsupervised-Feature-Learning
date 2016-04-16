function centroids = runkmeans(X, k, iterations)

  x2 = sum(X.^2,2);
  centroids = randn(k,size(X,2))*0.1;%X(randsample(size(X,1), k), :);
  BATCH_SIZE=1000;
  
  
  for itr = 1:iterations
    fprintf('K-means iteration %d / %d\n', itr, iterations);
    
    c2 = 0.5*sum(centroids.^2,2);

    summation = zeros(k, size(X,2));
    counts = zeros(k, 1);
    
    loss =0;
    
    for i=1:BATCH_SIZE:size(X,1)
      lastIndex=min(i+BATCH_SIZE-1, size(X,1));
      m = lastIndex - i + 1;
     
      [val,labels] = max(bsxfun(@minus,centroids*X(i:lastIndex,:)',c2));
      loss = loss + sum(0.5*x2(i:lastIndex) - val');
      
      S = sparse(1:m,labels,1,m,k,m); % labels as indicator matrix
      summation = summation + S'*X(i:lastIndex,:);
      counts = counts + sum(S,1)';
    end


    centroids = bsxfun(@rdivide, summation, counts);
    
    % just zap empty centroids so they don't introduce NaNs everywhere.
    badIndex = find(counts == 0);
    centroids(badIndex, :) = 0;
  end
