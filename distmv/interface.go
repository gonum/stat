// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package distmv

// A Rander can generate a random vector. If the input has length zero, then a
// new vector is allocated, otherwise the data will be stored in-place into
// the input.
type Rander interface {
	Rand([]float64) []float64
}

// A LogProber computes the log of the probability of the input vector.
type LogProber interface {
	LogProb(x []float64) float64
}
