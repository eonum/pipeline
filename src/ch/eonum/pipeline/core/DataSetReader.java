package ch.eonum.pipeline.core;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import ch.eonum.pipeline.distance.EuclidianDistance;
import ch.eonum.pipeline.util.Log;

/**
 * Abstract data set reader including prototype selection algorithms for certain two class problems.
 * #TODO move these prototype selection algorithms to a separate class.
 * 
 * @author tim
 *
 * @param <E>
 */
public abstract class DataSetReader<E extends Instance> extends Parameters implements DataPipeline<E> {
	protected File file;
	private boolean adjustRatio;
	/** prototype/training data selection mode when using ratio adjusting. */
	private ProtoTypeSelectionMode prototypeSelectionMode;
	
	public enum ProtoTypeSelectionMode {
		SPANNING, /** Spanning prototype selection SPS. */
		MAX_DISTANCE_TO_SMALL_CLASS, /** maximum distance to the center of the smaller class. */
		MIN_DISTANCE_TO_SMALL_CLASS, /** minimum distance to the center of the smaller class. */
		RANDOM, /** random selection. */
		CENTER, /** select prototypes in the center of the class. */
		BORDER, /** select prototypes at the border of the class. */
	}
	
	public DataSetReader(String fileName){
		this.file = new File(fileName);
		this.prototypeSelectionMode = ProtoTypeSelectionMode.RANDOM;
	}

	public DataSetReader(File file) {
		this.file = file;
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
		switch(this.prototypeSelectionMode){
		case SPANNING:
			return this.selectSpanningPrototypes(data);
		case MAX_DISTANCE_TO_SMALL_CLASS:
			return this.selectDistanceToCenterPrototypes(data, true, true);
		case MIN_DISTANCE_TO_SMALL_CLASS:
			return this.selectDistanceToCenterPrototypes(data, false, true);
		case RANDOM:
			return this.selectRandom(data);
		case CENTER:
			return this.selectDistanceToCenterPrototypes(data, false, false);
		case BORDER:
			return this.selectDistanceToCenterPrototypes(data, true, false);
		}
		return null;
	}

	private DataSet<E> selectRandom(DataSet<E> data) {
		DataSet<E> ds = new DataSet<E>();
		double ratio = data.getRatio();
		String largerClass = ratio > 1.0 ? "0" : "1";
		if(ratio > 1.0) ratio = 1.0/ratio;
		
		Random rand = new Random(234);

		for(E each : data)
			if(each.groundTruth.equals(largerClass))
				if(rand.nextDouble() < ratio)
					ds.addInstance(each);
			else
				ds.addInstance(each);
		return ds;
	}

	private DataSet<E> selectDistanceToCenterPrototypes(DataSet<E> data, final boolean max,
			boolean smallCenter) {
		DataSet<E> ds = new DataSet<E>();
		double ratio = data.getRatio();
		String largerClass = ratio > 1.0 ? "0" : "1";
		String smallerClass = ratio > 1.0 ? "1" : "0";
		Instance centerLargeClass = 
				new SparseInstance("large", largerClass, new HashMap<String, Double>());
		Instance centerSmallClass = 
				new SparseInstance("small", smallerClass, new HashMap<String, Double>());
		int large = 0;
		int small = 0;
		List<Instance> larger = new ArrayList<Instance>();
		for(E each : data)
			if(each.groundTruth.equals(largerClass)){
				larger.add(each);
				centerLargeClass.add(each);
				large++;
			} else {
				centerSmallClass.add(each);
				small++;
				ds.addInstance(each);
			} 
		centerLargeClass.divideBy((double)large);
		centerSmallClass.divideBy((double)small);
		
		/** select small instances from the large class. */
		EuclidianDistance<E> d = new EuclidianDistance<E>();
		ArrayList<E> distances = new ArrayList<E>();
		for(E each : data)
			if(each.groundTruth.equals(largerClass)){
				each.putResult("distance", d.distance(smallCenter ? centerSmallClass
						: centerLargeClass, each));
				distances.add(each);
			}
		Collections.sort(distances, new Comparator<Instance>(){
			@Override
			public int compare(Instance arg0, Instance arg1) {
				int c = ((Double)arg1.getResult("distance")).compareTo(arg0.getResult("distance"));
				return max ? c : -c;
			}});
		
		for(int i = 1; i < small; i ++){
			ds.addInstance(distances.get(i));
		}
		for(Instance i : data)
			i.removeResult("distance");
		for(Instance i : ds)
			i.removeResult("distance");

		return ds;
	}

	private DataSet<E> selectSpanningPrototypes(DataSet<E> data) {
		DataSet<E> ds = new DataSet<E>();
		double ratio = data.getRatio();
		String largerClass = ratio > 1.0 ? "0" : "1";
		int small = 0;
		List<E> larger = new ArrayList<E>();
		for(E each : data)
			if(each.groundTruth.equals(largerClass)){
				larger.add(each);

			} else {
				small++;
				ds.addInstance(each);
			} 

		EuclidianDistance<E> d = new EuclidianDistance<E>();

		ds.add(larger.get(0));
		Instance last = larger.get(0);
		larger.remove(0);
		for(Instance inst : larger)
			inst.putResult("distance", Double.MAX_VALUE);
		for(int i = 1; i < small; i ++){
			double max = 0;
			E maxInst = null;
			int index = 0;
			for(E inst : larger){
				double dist = d.distance(last, inst);
				if(inst.getResult("distance") > dist)
					inst.putResult("distance", dist);
				if(max < inst.getResult("distance")){
					max = inst.getResult("distance");
					maxInst = inst;
					index = larger.indexOf(inst);
				}
			}
			ds.add(maxInst);
			larger.remove(index);
			last = maxInst;
		}
		for(Instance i : data)
			i.removeResult("distance");
		for(Instance i : ds)
			i.removeResult("distance");

		return ds;
	}

	/**
	 * enable ratio adjusting. Remove some instances from the resulting data set
	 * until the ratio between "1" and "0" in a 2 class problem is equal to 1.
	 * The instances are removed according the selected prototypeSelectionMode.
	 * This makes sense for training sets with classifier which are sensitive to
	 * this ratio. Additionally we can speed up the training by removing the
	 * less important training instances.
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
		Log.error("A DataSetReader can only be at the begining of a pipeline");
	}
	
	@Override
	public void addInputTest(DataPipeline<E> input){
		Log.error("A DataSetReader can only be at the begining of a pipeline");
	}

}
