package ch.eonum.pipeline.reader;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import ch.eonum.pipeline.core.DataPipeline;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;
import ch.eonum.pipeline.filter.PrototypeSelection;
import ch.eonum.pipeline.filter.PrototypeSelection.ProtoTypeSelectionMode;
import ch.eonum.pipeline.util.Log;

/**
 * Abstract data set reader.
 * 
 * @author tim
 *
 * @param <E>
 */
public abstract class DataSetReader<E extends Instance> extends Parameters implements DataPipeline<E> {
	/** File to be read. */
	protected File file;
	/** Do ratio adjusting. {@link #enableRatioAdjusting} */
	private boolean adjustRatio;
	/** prototype selector for ratio adjustement. */
	private PrototypeSelection<E> prototypeSelector;
		
	public DataSetReader(String fileName){
		this.file = new File(fileName);
		this.prototypeSelector = new PrototypeSelection<E>(ProtoTypeSelectionMode.RANDOM);
	}

	public DataSetReader(File file) {
		this.file = file;
		this.prototypeSelector = new PrototypeSelection<E>(ProtoTypeSelectionMode.RANDOM);
	}

	/**
	 * read a data set from file.
	 * @param fileName
	 * @return
	 */
	public DataSet<E> readFromFile() {
		DataSet<E> data = new DataSet<E>();

		try {
			Scanner scanner = new Scanner(file);
					
			while (scanner.hasNext()) {
				convertLine(data, scanner.nextLine());
			}
			scanner.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		if(this.adjustRatio)
			return this.adjustRatio(data);
		return data;
	}
	
	private DataSet<E> adjustRatio(DataSet<E> data) {
		return this.prototypeSelector.ratioAdjust(data);
	}

	/**
	 * Enable ratio adjusting for two class problems. Remove some instances from
	 * the resulting data set until the ratio between "1" and "0" in a 2 class
	 * problem is equal to 1. The instances are removed according the selected
	 * prototypeSelectionMode. This makes sense for training sets with
	 * classifier which are sensitive to this ratio. Additionally we can speed
	 * up the training by removing the less important training instances.
	 * 
	 * @see ProtoTypeSelectionMode
	 * 
	 * @param data
	 */
	public void enableRatioAdjusting(){
		this.adjustRatio = true;
	}

	protected abstract void convertLine(DataSet<E> data, String line);
	
	/** pipeline methods . **/
	@Override
	public DataSet<E> trainSystem(boolean isResultDataSetNeeded){
		return this.readFromFile();
	}
	
	@Override
	public DataSet<E> testSystem(){
		return this.readFromFile();
	}
	
	@Override
	public void addInputTraining(DataPipeline<E> input){
		Log.error("A DataSetReader can only be at the start of a pipeline");
	}
	
	@Override
	public void addInputTest(DataPipeline<E> input){
		Log.error("A DataSetReader can only be at the start of a pipeline");
	}

}
