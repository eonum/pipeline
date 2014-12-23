package ch.eonum.pipeline.classification;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Jama.Matrix;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Log;

/**
 * Multiple linear regression including ridge regression.
 * 
 * @author tim
 *
 */
public class LinearRegression<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("lambda", "shrinkage parameter for ridge regression " +
				"(default 0.0 which is equal to linear regression)");
	}

	/** (#Features + 1) x 1 matrix Beta. Regression model. */
	private Matrix beta;
	/** norm beta vector to sum up to one. */
	private boolean normToOne;

	public LinearRegression(Features features){
		this.setFeatures(features);
		normToOne = false;
		this.setSupportedParameters(PARAMETERS);
		this.putParameter("lambda", 0.0);
		classify = false;
	}

	@Override
	public void train() {
		double lambda = this.getDoubleParameter("lambda");
		features = Features.removeConstantAndPerfectlyCorrellatedFeatures(features, trainingDataSet);
		int n = features.size() + 1;
		int m = this.trainingDataSet.size();
		Matrix Y = new Matrix(m, 1);
		Matrix X = new Matrix(m, n);
		int i = 0;
		for(Instance each : this.trainingDataSet){
			Y.set(i, 0, each.outcome);
			for(String feature : each.features())
				if(features.hasFeature(feature)) 
					X.set(i, features.getIndexFromFeature(feature) + 1, each.get(feature));
			X.set(i,0,1.0);
			i++;
		}
		try {
			Matrix XtX1 = (X.transpose().times(X));//
			/** ridge regression. */
			if(lambda != 0)
				XtX1 = XtX1.plus(Matrix.identity(n, n).times(lambda));
			XtX1 = XtX1.inverse();
			this.beta = XtX1.times(X.transpose()).times(Y);
		} catch (Exception e) {
			Log.warn(e.getMessage());
			Log.warn("Linear Regression: Beta cannot be calculated");
			e.printStackTrace();
			this.beta = new Matrix(n, 1);
		}
		if(normToOne){
			double total = 0.0;
			for(i = 0; i < beta.getRowDimension(); i++)
				total += beta.get(i, 0);
			for(i = 0; i < beta.getRowDimension(); i++)
				beta.set(i, 0, beta.get(i, 0)/total);
		}
		this.save(baseDir + "matrixBeta");
	}

	@Override
	public DataSet<E> test(){
		this.loadSerializedState(new File(baseDir + "matrixBeta"));
		for(Instance each : this.testDataSet){
			each.putResult("result", 0.);
			double prediction = this.beta.get(0, 0);
			for(String feature : each.features())
				if(features.hasFeature(feature))
					prediction += this.beta.get(
							features.getIndexFromFeature(feature) + 1, 0)
							* each.get(feature);
			each.putResult("result", Math.max(0, prediction));
		}
		return this.testDataSet;
	}
	
	@Override
	public void save(String fileName) {
		try {
			PrintStream p = new PrintStream(new FileOutputStream(fileName));
			for(int x = 0; x < this.beta.getColumnDimension(); x++){
				for(int y = 0; y < this.beta.getRowDimension(); y++)
					p.print(this.beta.get(y,x) + " ");
				p.println();
			}
			p.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void loadSerializedState(File file){
		try {
			FileInputStream fstream = new FileInputStream(file);
			BufferedReader br = new BufferedReader(
					new InputStreamReader(fstream));
			String line;
			List<Double[]> columns = new ArrayList<Double[]>();
			while ((line = br.readLine()) != null) {
				String[] row = line.split(" ");
				Double[] rowI = new Double[row.length];
				for(int i = 0; i < row.length; i++)
					rowI[i] = Double.parseDouble(row[i]);
				columns.add(rowI);
			}
			this.beta = new Matrix(columns.get(0).length, columns.size());
			for(int x = 0; x < columns.get(0).length; x++)
				for(int y = 0; y < columns.size(); y++)
					this.beta.set(x, y, columns.get(y)[x]);
			br.close();
 
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void normToOne() {
		this.normToOne = true;
	}

}
