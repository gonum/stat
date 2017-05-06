package distmv

import (
	"math"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

type DistNormaler interface {
	DistNormal(l, r *Normal) float64
}

type DistUniformer interface {
	DistUniform(l, r *Uniform) float64
}

// Bhattacharyya is a type for computing the Bhattacharyya distance between
// probability distributions.
//
// The Battachara distance is defined as
//  D_B = -ln(BC(l,r))
//  BC = \int_x (p(x)q(x))^(1/2) dx
// Where BC is known as the Bhattacharyya coefficient.
// The Bhattacharyya distance is related to the Hellinger distance by
//  H = sqrt(1-BC)
// For more information, see
//  https://en.wikipedia.org/wiki/Bhattacharyya_distance
type Bhattacharyya struct{}

// DistNormal computes the Bhattacharyya distance between normal distributions l and r.
// The dimensions of the input distributions must match or DistNormal will panic.
//
// For Normal distributions, the Bhattacharyya distance is
//  Σ = (Σ_l + Σ_r)/2
//  D_B = (1/8)*(μ_l - μ_r)^T*Σ^-1*(μ_l - μ_r) + (1/2)*ln(det(Σ)/(det(Σ_l)*det(Σ_r))^(1/2))
func (Bhattacharyya) DistNormal(l, r *Normal) float64 {
	dim := l.Dim()
	if dim != r.Dim() {
		panic(badSizeMismatch)
	}

	var sigma mat64.SymDense
	sigma.AddSym(&l.sigma, &r.sigma)
	sigma.ScaleSym(0.5, &sigma)

	var chol mat64.Cholesky
	chol.Factorize(&sigma)

	mahalanobis := stat.Mahalanobis(mat64.NewVector(dim, l.mu), mat64.NewVector(dim, r.mu), &chol)
	mahalanobisSq := mahalanobis * mahalanobis

	dl := l.chol.LogDet()
	dr := r.chol.LogDet()
	ds := chol.LogDet()

	return 0.125*mahalanobisSq + 0.5*ds - 0.25*dl - 0.25*dr
}

// DistUniform computes the Bhattacharyya distance between uniform distributions l and r.
// The dimensions of the input distributions must match or DistUnif will panic.
func (Bhattacharyya) DistUniform(l, r *Uniform) float64 {
	if len(l.bounds) != len(r.bounds) {
		panic(badSizeMismatch)
	}
	// BC = \int \sqrt(p(x)q(x)), which for uniform distributions is a constant
	// over the volume where both distributions have positive probability.
	// Compute the overlap and the value of sqrt(p(x)q(x)). The entropy is the
	// negative log probability of the distribution (use instead of LogProb so
	// it is not necessary to construct an x value).
	//
	// BC = volume * sqrt(p(x)q(x))
	// logBC = log(volume) + 0.5*(logP + logQ)
	// D_B = -logBC
	return -unifLogVolOverlap(l.bounds, r.bounds) + 0.5*(l.Entropy()+r.Entropy())
}

// unifLogVolOverlap computes the log of the volume of the hyper-rectangle where
// both uniform distributions have positive probability.
func unifLogVolOverlap(b1, b2 []Bound) float64 {
	var logVolOverlap float64
	for dim, v1 := range b1 {
		v2 := b2[dim]
		// If the surfaces don't overlap, then the volume is 0
		if v1.Max <= v2.Min || v2.Max <= v1.Min {
			return math.Inf(-1)
		}
		vol := math.Min(v1.Max, v2.Max) - math.Max(v1.Min, v2.Min)
		logVolOverlap += math.Log(vol)
	}
	return logVolOverlap
}

// KullbackLiebler is a type for computing the Kullback-Leibler divergence from l to r.
// The dimensions of the input distributions must match or the function will panic.
//
// The Kullback-Liebler divergence is defined as
//  D_KL(l || r ) = \int_x p(x) log(p(x)/q(x)) dx
// Note that the Kullback-Liebler divergence is not symmetric with respect to
// the order of the input arguments.
type KullbackLeibler struct{}

// DistNormal returns the KullbackLeibler distance between normal distributions l and r.
// The dimensions of the input distributions must match or DistNormal will panic.
//
// For two normal distributions, the KL divergence is computed as
//   D_KL(l || r) = 0.5*[ln(|Σ_r|) - ln(|Σ_l|) + (μ_l - μ_r)^T*Σ_r^-1*(μ_l - μ_r) + tr(Σ_r^-1*Σ_l)-d]
func (KullbackLeibler) DistNormal(l, r *Normal) float64 {
	dim := l.Dim()
	if dim != r.Dim() {
		panic(badSizeMismatch)
	}

	mahalanobis := stat.Mahalanobis(mat64.NewVector(dim, l.mu), mat64.NewVector(dim, r.mu), &r.chol)
	mahalanobisSq := mahalanobis * mahalanobis

	// TODO(btracey): Optimize where there is a SolveCholeskySym
	// TODO(btracey): There may be a more efficient way to just compute the trace
	// Compute tr(Σ_r^-1*Σ_l) using the fact that Σ_l = U^T * U
	var u mat64.TriDense
	u.UFromCholesky(&l.chol)
	var m mat64.Dense
	err := m.SolveCholesky(&r.chol, u.T())
	if err != nil {
		return math.NaN()
	}
	m.Mul(&m, &u)
	tr := mat64.Trace(&m)

	return r.logSqrtDet - l.logSqrtDet + 0.5*(mahalanobisSq+tr-float64(l.dim))
}

// DistUniform returns the KullbackLeibler distance between normal distributions l and r.
// The dimensions of the input distributions must match or DistNormal will panic.
func (KullbackLeibler) DistUniform(l, r *Uniform) float64 {
	bl := l.Bounds(nil)
	br := r.Bounds(nil)
	if len(bl) != len(br) {
		panic(badSizeMismatch)
	}

	// The KL is ∞ if x1 is not completely contained within x2, because then
	// r(x) is zero when l(x) is non-zero.
	contained := true
	for i, v := range bl {
		if br[i].Min > v.Min || br[i].Max < v.Max {
			contained = false
			break
		}
	}
	if !contained {
		return math.Inf(1)
	}

	// The KL divergence is finite.
	//
	// KL defines 0*ln(0) = 0, so there is no contribution to KL where l(x) = 0.
	// Inside the region, l(x) and r(x) are constant (uniform distribution), and
	// this constant is integrated over l(x), which integrates out to one.
	// The entropy is -log(p(x)).
	logPx := -l.Entropy()
	logQx := -r.Entropy()
	return logPx - logQx
}
