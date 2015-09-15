package ch.eonum.pipeline.validation;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.eonum.pipeline.core.DataPipeline;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.core.Parameters;
import ch.eonum.pipeline.evaluation.Evaluator;
import ch.eonum.pipeline.util.FileUtil;
import ch.eonum.pipeline.util.Gnuplot;
import ch.eonum.pipeline.util.Log;

/**
 * evaluation of a whole system, a whole pipeline of classifiers, extractors and readers.
 * 
 * @author tim
 *
 */
public class SystemValidator<E extends Instance> {
	private DataPipeline<E> system;
	private Evaluator<E> evaluator;
	private String baseDir;
	
	public SystemValidator(DataPipeline<E> system, Evaluator<E> evaluator){
		this.system = system;
		this.evaluator = evaluator;
	}
	
	public double evaluate(){
		return this.evaluate(false, null);
	}
	
	public double evaluate(boolean printResults, String name){
		system.trainSystem(false);
		DataSet<E> test = system.testSystem();
		double ret = evaluator.evaluate(test);
		if(printResults)
			evaluator.printResultsAndGnuplot(this.getBaseDir() + name);
		return ret;
	}

	public void setBaseDir(String baseDir) {
		this.baseDir = baseDir;
	}

	public String getBaseDir() {
		return baseDir;
	}

	/**
	 * Validate a system along a range of a certain parameter and return the value
	 * of the parameter where the optimum is reached.
	 * 
	 * @param parameters Parameters objects, which have to be adjusted
	 * @param param Parameter which is to be validated
	 * @param start start of range
	 * @param stop end of range
	 * @param step step size
	 * @param logSpace 
	 * @param parentResultsFolder 
	 * @return parameter optimum
	 */
	public double validateParameter(Parameters[] parameters, String param,
			double start, double stop, double step) {
		return this.validateParameter(parameters, param, start, stop, step, null, false);
	}
	
	/**
	 * Validate a system along a range of a certain parameter and return the value
	 * of the parameter where the optimum is reached.
	 * This method additionally outputs all values and gnuplots them.
	 * 
	 * @param parameters Parameters objects, which have to be adjusted
	 * @param param Parameter which is to be validated
	 * @param start start of range
	 * @param stop end of range
	 * @param step stepsize
	 * @param parentResultsFolder parent result folder
	 * @return parameter optimum
	 */
	public double validateAndGnuplotParameter(Parameters[] parameters,
			String param, Double start, Double stop, double step,
			String parentResultsFolder) {
		return this.validateParameter(parameters, param, start, stop, step, parentResultsFolder, false);
	}
	
	/**
	 * Validate a system along a range in log2 space of a certain parameter and return the value
	 * of the parameter where the optimum is reached.
	 * This method additionally outputs all values and gnuplots them.
	 * 
	 * @param parameters Parameters objects, which have to be adjusted
	 * @param param Parameter which is to be validated
	 * @param start start of range in log2 space
	 * @param stop end of range in log2 space
	 * @param step step size in log2 space
	 * @param parentResultsFolder parent result folder
	 * @return parameter optimum in real space
	 */
	public double validateAndGnuplotParameterLogSpace(Parameters[] parameters,
			String param, Double start, Double stop, double step,
			String parentResultsFolder) {
		return this.validateParameter(parameters, param, start, stop, step, parentResultsFolder, true);
	}

	private double validateParameter(Parameters[] parameters, String param,
			double start, double stop, double step, String parentResultsFolder, boolean logSpace) {
		int istart = (int)(start/step);
		int istop = (int)(stop/step);
		
		double maxParam = start;
		double maxValue = Double.NEGATIVE_INFINITY;
		
		List<Double> parameterValues = new ArrayList<Double>();
		Map<Double, Double> parameterMeasures = new HashMap<Double, Double>();
		
		for(int i = istart; i <= istop; i++){
			double p = i*step;
			double pReal = p;
			if(logSpace)
				pReal = Math.pow(2, p);
			for(Parameters each : parameters)
				each.putParameter(param, pReal);
			double value = this.evaluate();
			parameterValues.add(p);
			parameterMeasures.put(p, value);
			System.out.println((p) + " = " + value);
			if(value > maxValue){
				maxValue = value;
				maxParam = pReal;
			}
		}
		for(Parameters each : parameters)
			each.putParameter(param, maxParam);
		
		if(parentResultsFolder != null){
			FileUtil.mkdir(parentResultsFolder + param);
			try {
				FileWriter fstream = new FileWriter(parentResultsFolder + param + "/validation.txt");
				BufferedWriter out = new BufferedWriter(fstream);
				for(Double each : parameterValues){
					out.write(each + "	" + parameterMeasures.get(each) + "\n");
				}
				out.close();
			} catch (Exception e) {
				System.err.println("Error: " + e.getMessage());
			}
			Gnuplot.plotOneDimensionalCurve(parentResultsFolder + param + "/validation.txt", "Validation of " + param);
		}
		return maxParam;
	}

	/**
	 * Find out the local optimum for all meta parameters. 
	 * @param params
	 * @param nIterations
	 * @param dir directory for results
	 * @return
	 */
	public Map<ParameterValidation, Double> parameterSearch(
			List<ParameterValidation> params, int nIterations, String dir) {
		
		Map<ParameterValidation, Double> values = new HashMap<ParameterValidation, Double>();
		for(ParameterValidation each : params)
			values.put(each, each.getInitialvalue());
		
		FileUtil.mkdir(dir);
		
		try {
			PrintStream p = new PrintStream(new FileOutputStream(dir
					+ "parameterSearch.txt"));

			for(int i = 1; i <= nIterations; i++){
				boolean changed = false;
				
				Log.puts("Iteration " + i);
				p.println("Iteration " + i);
				for (ParameterValidation each : params) {
					double newValue = this.validateParameter(
							each.getParameters(), each.getName(), each.getStart(),
							each.getEnd(), each.getStepSize(), dir, each.isLog2Space());
					if(each.isLog2Space())
						newValue = Math.log(newValue)/Math.log(2);
					
					if(Math.abs(newValue - values.get(each)) > 0.0000001){
						changed = true;
						values.put(each, newValue);
					}
					System.out.println("Maximum for " + each.getName() + ": " + newValue);
					p.println("Maximum for " + each.getName() + ": " + newValue);
				}
	
				// check if something has changed. when nothing changes we can finish
				if(!changed)
					break;
	
				// adapt ranges
				for(ParameterValidation each : params){
					double newStart = values.get(each) - each.getRange()/2.0;
					each.setStart(newStart > each.getMin() ? newStart : each.getMin());
					
					double newEnd = values.get(each) + each.getRange()/2.0;
					each.setEnd(newEnd < each.getMax() ? newEnd : each.getMax());
				}
			}
			p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return values;
	}
	
	/**
	 * find out the optimum for two parameters using an exhaustive grid search.
	 * @param param1 parameter validation 1
	 * @param param2 parameter validation 2
	 * @param dir directory for results
	 * @return
	 */
	public Map<ParameterValidation, Double> gridSearch(
			ParameterValidation param1, ParameterValidation param2, String dir) {
		
		Map<ParameterValidation, Double> values = new HashMap<ParameterValidation, Double>();
		values.put(param1, param1.getInitialvalue());
		values.put(param2, param2.getInitialvalue());
		
		FileUtil.mkdir(dir);
		
		try {
			PrintStream p = new PrintStream(new FileOutputStream(dir
					+ "grid.txt"));
			
			double max = Double.NEGATIVE_INFINITY;

			int numSteps1 = (int)(param1.getRange()/param1.getStepSize());
			for(int p1 = 0; p1 <= numSteps1; p1++){
				double p1d = param1.getStart() + p1 * param1.getStepSize();
				int numSteps2 = (int)(param2.getRange()/param2.getStepSize());
				for(int p2 = 0; p2 <= numSteps2; p2++){
					double p2d = param2.getStart() + p2 * param2.getStepSize();
					double p1r = p1d;
					if(param1.isLog2Space()) p1r = Math.pow(2.0, p1d);
					double p2r = p2d;
					if(param2.isLog2Space()) p2r = Math.pow(2.0, p2d);
					for(Parameters each : param1.getParameters())
						each.putParameter(param1.getName(), p1r);
					for(Parameters each : param2.getParameters())
						each.putParameter(param2.getName(), p2r);
					double m = this.evaluate();
					if(m > max){
						max = m;
						values.put(param1, p1d);
						values.put(param2, p2d);
					}
					Log.puts("[" + param1.getName() + " = " + p1d + ", " + param2.getName() + " = " + p2d + "] => " + m);
					p.println(p1d + " " + p2d + " " + m);
				}
			}
			
			p.close();
			Gnuplot.plotGrid(dir + "grid.txt", "grid");
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return values;
	}
}
