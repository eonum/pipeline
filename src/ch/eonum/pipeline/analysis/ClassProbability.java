package ch.eonum.pipeline.analysis;

/**
 * A class and it's probability for a certain test sample.
 * @author tim
 *
 */
public final class ClassProbability implements Comparable<ClassProbability> {

	public final String className;
	public final double prob;

	public ClassProbability(String className, double prob) {
		this.className = className;
		this.prob = prob;
	}

	@Override
	public int compareTo(ClassProbability cp2) {
		if(this.prob == cp2.prob) return 0;
		return (this.prob - cp2.prob > 0.0) ? 1 : -1;
	}
	
}