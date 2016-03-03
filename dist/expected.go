// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dist

import "github.com/gonum/quad"

// ExpectedFixed numerically computes the expected value of the input function
// under the probability distribution specified by the Quantiler. That
// is, ExpectedFixed computes
//
//  ev ≈ E[f] = int_-inf^inf f(x) p(x) dx
//
// using the number of quadrature locations specified by evals. Concurrent
// sets the number of allowed concurrent evaluations of f.
//
// ExpectedFixed is a wrapper around quad.Fixed. See the documentation for more
// information.
func ExpectedFixed(f func(float64) float64, q Quantiler, evals, concurrent int) (ev float64) {
	// Rather than integrate the documentation integral directly, instead transform
	// into a more numerically favorable integral.
	//  E[f] = int_-inf^inf f(x) p(x) dx
	//       = int_0^1 f(icdf(x')) dx'
	// Proof: Integrate by substitution https://en.wikipedia.org/wiki/Integration_by_substitution.
	//  int_phi(a)^phi(b) = int_a^b f(phi(t)) phi'(t) dt
	// Let phi(t) = cdf(t)
	//  E[f] = int_0^1 f(icdf(x')) dx'
	//       = int_cdf(-inf)^cdf(inf) f(icdf(x')) dx
	// Let phi(t) = cdf(t), and integrate by substitution
	// (https://en.wikipedia.org/wiki/Integration_by_substitution.):
	//  E[f] = int_-inf^inf f(cdf(icdf(x))) cdf'(x) dx
	//       = int_-inf^inf f(x) pdf(x) dx
	// because cdf(icdf(x)) = x, and the probability density is the derivative
	// of the cumuluative distribution
	//
	// This transformed integral should converge faster (in terms of quadrature
	// locations) than the original integral since it focuses the evaluations
	// in the plausible range of x rather than evaluating lots of locations where
	// p(x) ≈ 0. Additionally, the original integral cannot be computed numercally
	// for discrete distributions, as p(x) is a series of 𝛿-functions, but can
	// when using the quantile.

	// TODO(btracey): Use a type switch on the quantiler when quad supports
	// more quadrature classes. For example, use Gaussian-Hermite quadrature
	// for the normal distribution.
	g := func(p float64) float64 {
		x := q.Quantile(p)
		return f(x)
	}
	return quad.Fixed(g, 0, 1, evals, nil, concurrent)
}
