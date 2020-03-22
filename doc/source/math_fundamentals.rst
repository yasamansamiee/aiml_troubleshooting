Mathematical fundamentals
=========================

This section is concerned with the mathematical derivation
of the formulas in `(Kimura et al. 2011)`_.

* Likelihood :math:`f(x,y|\Phi)`:

    .. math::

        \begin{array}{rcl}
        \log f(x,y|\Phi)
        &=& \log \left( \pi_x \cdot \mathcal N(m_x,\Sigma_x) \right) \\
        &=& \log \left( \pi_x \cdot  \frac{1}{ \sqrt{(2\pi)^n |\Sigma|}} \exp \left(\frac {1}{2} (y - m_x)^T \Sigma^{-1}_x (y - m_x) \right)
        \right) \\
        &=& \log \pi_x - \frac{n}{2} \log(2\pi) - \frac{1}{2} \log |\Sigma| - \frac{1}{2} (y - m_x)^T \Sigma^{-1}_x (y - m_x) \\
        &=&  - \frac{n}{2} \log(2\pi) + \sum_{i=1}^k \left( \log \pi_i - \frac{1}{2} \log |\Sigma_i| \right. \\
        && - \frac{1}{2} y^T \Sigma^{-1}_i y - \frac{1}{2} m_i^T \Sigma^{-1}_i m_i
            \left. -  y^T \Sigma^{-1}_i m_i \right)\cdot \delta_i(x), \quad \text{where} \\
            %
        \delta_i(x) &=& \begin{cases} 1, & x=i \\ 0, & \text{else} \end{cases}
        \end{array} $$

* penalized likelihood

    .. math::

        \begin{array}{rcl}
            ~\phi^{(j)} &=&\arg \max_\Phi Q(\phi,\phi^{(j-1)}) \\
        Q(\phi,\phi^{(j-1)})
        &=& \mathbb E [ \log f(x_{1:T},y_{1:T}|\Phi )| y_{1:T}, \Phi^{(j-1)}]\\
            &=& \sum^T_{t=1} \sum^k_{i=1} \left( \log \pi_i - \frac{1}{2} \log |\Sigma_i| \right. \\
        && - \frac{1}{2} y_t^T \Sigma^{-1}_i y_t - \frac{1}{2} m_i^T \Sigma^{-1}_i m_i
            \left. -  y_t^T \Sigma^{-1}_i m_i \right) \cdot Pr(x_t=i | y_t, \Phi^{(j-1)}) \\
        Pr(x_t=i | y_t, \Phi^{(j-1)})
        &=& \frac{ \pi_i \cdot \mathcal N \left(y_t; m_i^{(j-1)},\Sigma_i^{(j-1)}\right)}{\sum_{l=1}^k \pi_l \cdot \mathcal N \left(y_t; m_l^{(j-1)},\Sigma_l^{(j-1)}\right)}
        \end{array}

* Optimization step in EM algorithm

    .. math::

        \begin{array}{rcl}
            ~\phi^{(j)} &=&\arg \max_\Phi Q(\phi,\phi^{(j-1)}) \\
        Q(\phi,\phi^{(j-1)})
        &=& \mathbb E [ \log f(x_{1:T},y_{1:T}|\Phi )| y_{1:T}, \Phi^{(j-1)}]\\
        &=& \mathbb E [ \log \prod_{t=1}^T f(x_t,y_t|\Phi )| y_{1:T}, \Phi^{(j-1)}]\\
        &=& \sum^T_t \mathbb E [ \log f(x_t,y_t|\Phi )| y_{1:T}, \Phi^{(j-1)}] & \prod \text{ becomes } \sum \\
        &=& \sum^T_t \sum_i^k \big( \log f(i,y_t|\Phi ) \cdot Pr(x_t=i | y_t, \Phi^{(j-1)}) \big) \\
            &=& \sum^T_{t=1} \sum^k_{x=1} \sum^k_{i=1}\left( \log \pi_i - \frac{1}{2} \log |\Sigma_i| \right. \\
        && - \frac{1}{2} y_t^T \Sigma^{-1}_i y_t - \frac{1}{2} m_i^T \Sigma^{-1}_i m_i
            \left. -  y_t^T \Sigma^{-1}_i m_i \right) \delta_i(x)\cdot Pr(x_t=i | y_t, \Phi^{(j-1)}) \\
        &=& \sum^T_{t=1} \sum^k_{i=1} \left( \log \pi_i - \frac{1}{2} \log |\Sigma_i| \right. \\
        && - \frac{1}{2} y_t^T \Sigma^{-1}_i y_t - \frac{1}{2} m_i^T \Sigma^{-1}_i m_i
            \left. -  y_t^T \Sigma^{-1}_i m_i \right) \cdot Pr(x_t=x | y_t, \Phi^{(j-1)})
        \end{array}

* Computing the responsibilities is the Expectation step.

    .. math::

        \begin{array}{rcl}
        Pr(x_t=i | y_t, \Phi^{(j-1)})
        &=& \frac{ \pi_i \cdot \mathcal N \left(y_t; m_i^{(j-1)},\Sigma_i^{(j-1)}\right)}{\sum_{l=1}^k \pi_l \cdot \mathcal N \left(y_t; m_l^{(j-1)},\Sigma_l^{(j-1)}\right)}
        \end{array}

* The maximizing values are then (TODO check the calculus)

    .. math::

        \begin{array}{rcl}
        ~\pi_i^{(j)}
        &=& \frac{\sum^t_{t=1} \mathbb E[\delta_i(x_t)|y_{1:T},\Phi^{(j-1)}]} {T} \\
        &=& \frac{\sum^t_{t=1} Pr(x_t=i | y_t, \Phi^{(j-1)})}{T}\\
        ~m_i^{(j)}
        &=& \frac{\sum^t_{t=1} \mathbb E[y_t \cdot \delta_i(x_t)|y_{1:T},\Phi^{(j-1)}]} {\sum^t_{t=1} \mathbb E[\delta_i(x_t)|y_{1:T},\Phi^{(j-1)}]} \\
        &=& \frac{\sum^t_{t=1} y_t \cdot Pr(x_t=i | y_t, \Phi^{(j-1)})}{T\cdot \pi_i^{(j)}}\\
        \Sigma_i^{(j)}
        &=& \frac{\sum^t_{t=1} \mathbb E[y_t \cdot y_t^T \cdot \delta_i(x_t)|y_{1:T},\Phi^{(j-1)}]}
        {\sum^t_{t=1} \mathbb E[\delta_i(x_t)|y_{1:T},\Phi^{(j-1)}]} - m_i^{(j)} \cdot m_i^{(j)T}\\
        &=& \frac{\sum^t_{t=1} y_t\cdot y_t^T \cdot Pr(x_t=i | y_t, \Phi^{(j-1)})}{T\cdot \pi_i^{(j)}} - m_i^{(j)}\cdot m_i^{(j)T}\\
        \end{array}


.. _(Kimura et al. 2011): https://link.springer.com/article/10.1007/s10044-011-0256-4
