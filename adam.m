function theta = adam(gradfunc, ini_theta, options)
if ~isfield(options, 'alpha')
    alpha = 1e-3;
else
    alpha = options.alpha;
end
if ~isfield(options, 'beta1')
    beta1 = .9;
else
    beta1 = options.beta1;
end
if ~isfield(options, 'beta2')
    beta2 = .999;
else
    beta2 = options.beta2;
end
if ~isfield(options, 'epsilon')
    epsilon = 1e-8;
else
    epsilon = options.epsilon;
end
if ~isfield(options, 'max_iter')
    max_iter = 500;
else
    max_iter = options.max_iter;
end

t = 0;
theta0 = ini_theta;
m0 = zeros(size(theta0));
v0 = diag(m0);

theta_old = theta0;
m_old = m0;
v_old = v0;

while t <= max_iter
    t = t + 1;
    gt = gradfunc(theta_old);
    mt = beta1*m_old + (1-beta1)*gt;
    vt = beta2*v_old + (1-beta2)*diag(gt.^2);
    unbiased_mt = mt/(1-beta1^t);
    unbiased_vt = vt/(1-beta2^t);
    thetat = theta_old - alpha*unbiased_mt/(sqrtm(unbiased_vt) + epsilon);
    theta_old = thetat;
    m_old = mt;
    v_old = vt;
end
theta = thetat;
end
