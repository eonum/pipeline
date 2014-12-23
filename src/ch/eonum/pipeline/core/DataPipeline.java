package ch.eonum.pipeline.core;

/**
 * interface for all objects which can be part of a data pipeline with at least
 * a data set input slot or a data set output slot.
 * @author tim
 *
 */
public interface DataPipeline<E extends Instance> {

	/**
	 * train/prepare/read/extract
	 * @return
	 */
	public DataSet<E> trainSystem(boolean isResultDataSetNeeded);

	/**
	 * test/read/extract
	 * @return
	 */
	public DataSet<E> testSystem();
	
	/**
	 * connect the training data input connector.
	 * @param input
	 */
	public void addInputTraining(DataPipeline<E> input);
	
	/**
	 * connect the test data input connector.
	 * @param input
	 */
	public void addInputTest(DataPipeline<E> input);

}
