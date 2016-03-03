// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dist

import (
	"math"
	"testing"

	"github.com/gonum/floats"
)

func TestExpected(t *testing.T) {
	for cas, test := range []struct {
		f     func(float64) float64
		q     Quantiler
		evals int
		ev    float64
		tol   float64
	}{
		{
			// Expected value of unit normal.
			f:     func(x float64) float64 { return x },
			q:     UnitNormal,
			evals: 10,
			ev:    0,
			tol:   1e-10,
		},
		{
			// Variance of unit normal loose tolerance.
			f:     func(x float64) float64 { return x * x },
			q:     UnitNormal,
			evals: 100,
			ev:    1,
			tol:   1e-3,
		},
		{
			// Variance of unit normal tight tolerance.
			f:     func(x float64) float64 { return x * x },
			q:     UnitNormal,
			evals: 1000000,
			ev:    1,
			tol:   1e-10,
		},
		{
			// Expected value of non-unit normal.
			f:     func(x float64) float64 { return x },
			q:     Normal{Mu: 5, Sigma: 10},
			evals: 100,
			ev:    5,
			tol:   1e-10,
		},
		{
			// Variance of non-unit normal loose tolerance.
			f:     func(x float64) float64 { return (x - 5) * (x - 5) },
			q:     Normal{Mu: 5, Sigma: 10},
			evals: 100,
			ev:    100,
			tol:   1e-3,
		},
		{
			// Variance of non-unit normal tight tolerance.
			f:     func(x float64) float64 { return (x - 5) * (x - 5) },
			q:     Normal{Mu: 5, Sigma: 10},
			evals: 1000000,
			ev:    100,
			tol:   1e-10,
		},
		{
			// 5^th moment of exponential loose tolerance.
			f:     func(x float64) float64 { return math.Pow(x, 5) },
			q:     Exponential{Rate: 3},
			evals: 1000,
			ev:    120.0 / 243.0,
			tol:   1e-3,
		},
		{
			// 5^th moment of exponential tight tolerance.
			f:     func(x float64) float64 { return math.Pow(x, 5) },
			q:     Exponential{Rate: 3},
			evals: 1000000,
			ev:    120.0 / 243.0,
			tol:   1e-8,
		},
	} {
		ev := ExpectedFixed(test.f, test.q, test.evals, 0)
		if !floats.EqualWithinAbsOrRel(ev, test.ev, test.tol, test.tol) {
			t.Errorf("Cas %v. Expected value mismatch. Want %v, got %v", cas, test.ev, ev)
		}
	}
}
