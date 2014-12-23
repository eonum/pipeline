package ch.eonum.pipeline.transformation;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Set;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.SparseInstance;

/**
 * normalize a data set to fit into the interval [0,1].
 * 
 * Note: Dimensions/Features not present in the training set will not be normalized at all.
 * 
 * 
 * @author tim
 *
 */
public class MinMaxNormalizer<E extends Instance> extends Transformer<E> {
	private SparseInstance max;
	private SparseInstance min;
	private Instance delta;
	private boolean verbose = false;

	/**
	 * create a normalizer from a given dataset.
	 * max and min are calculated out of the dataset.
	 * @param dataset
	 */
	public MinMaxNormalizer(DataSet<E> dataset) {
		this.prepare(dataset);
	}
	
	@Override
	public void prepare(DataSet<E> dataset){
		max = new SparseInstance("", "", new HashMap<String, Double>());
		min = new SparseInstance("", "", new HashMap<String, Double>());
		Set<String> features = dataset.features();
		for(Instance each : dataset)
			for(String feature : features){
				if(max.get(feature) < each.get(feature) || !max.features().contains(feature))
					max.put(feature, each.get(feature));
				if(min.get(feature) > each.get(feature) || !min.features().contains(feature))
					min.put(feature, each.get(feature));
			}
		min.cleanUp();
		max.cleanUp();
		calculateDelta();
	}
	
	/**
	 * create a normalizer by loading max and min from file.
	 * format: one line for each feature: 'feature minValue maxValue'
	 * @param fileName
	 */
	public MinMaxNormalizer(String fileName) {
		max = new SparseInstance("", "", new HashMap<String, Double>());
		min = new SparseInstance("", "", new HashMap<String, Double>());
		try {
			FileInputStream fstream = new FileInputStream(fileName);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			while ((strLine = br.readLine()) != null) {
				String[] values = strLine.split(" ");
				min.put(values[0], Double.valueOf(values[1]));
				max.put(values[0], Double.valueOf(values[2]));
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		calculateDelta();
	}

	private void calculateDelta() {
		delta = max.minusStateless(min);
		if(verbose ) System.out.println("minimum instance" + min);
		if(verbose) System.out.println("maximum instance" + max);
		delta.cleanUp();
	}

	@Override
	public void extract(){
		super.extract();
		for(Instance each : this.dataSet){
			each.minus(min);
			each.divideBy(delta);
			each.cleanUp();
		}
	}

	public void writeToFile(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			for(String feature : max.features())
				p.println(feature + " " + min.get(feature) + " " + max.get(feature));
			p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void setVerbose(){
		this.verbose = true;
	}

}
