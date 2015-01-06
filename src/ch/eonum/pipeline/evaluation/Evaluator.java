package ch.eonum.pipeline.evaluation;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;

/**
 * Interface for all classes in this package. In many classifiers (especially
 * meta classifiers) and almost all optimizers, an evaluator has to be provided.
 * 
 * @author tim
 * 
 * @param <E>
 */
public interface Evaluator<E extends Instance> {
	/**
	 * evaluate a data set and return the main measure. The higher the evaluation
	 * measure, the better. Optimization algorithms maximize this measure. If
	 * you have a measure to be minimized, return the negative value.
	 * 
	 * @param dataset
	 * @return evaluation measure
	 */
	public double evaluate(DataSet<E> dataset);
	/**
	 * print the measurement with optional measurements to file.
	 * @param fileName
	 */
	public void printResults(String fileName);
	/**
	 * execute printResults and gnuplot the results in the same folder.
	 * @param fileName
	 */
	public void printResultsAndGnuplot(String fileName);
}
