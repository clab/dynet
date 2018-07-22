import static edu.cmu.dynet.internal.dynet_swig.as_scalar;
import static edu.cmu.dynet.internal.dynet_swig.as_vector;
import static edu.cmu.dynet.internal.dynet_swig.exprPlus;
import static edu.cmu.dynet.internal.dynet_swig.exprTimes;
import static edu.cmu.dynet.internal.dynet_swig.initialize;
import static edu.cmu.dynet.internal.dynet_swig.lookup;
import static edu.cmu.dynet.internal.dynet_swig.parameter;
import static edu.cmu.dynet.internal.dynet_swig.pickneglogsoftmax;
import static edu.cmu.dynet.internal.dynet_swig.softmax;
import static edu.cmu.dynet.internal.dynet_swig.sum;
import static edu.cmu.dynet.internal.dynet_swig.sum_batches;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import edu.cmu.dynet.internal.ComputationGraph;
import edu.cmu.dynet.internal.Dim;
import edu.cmu.dynet.internal.DynetParams;
import edu.cmu.dynet.internal.Expression;
import edu.cmu.dynet.internal.ExpressionVector;
import edu.cmu.dynet.internal.FloatVector;
import edu.cmu.dynet.internal.LongVector;
import edu.cmu.dynet.internal.LookupParameter;
import edu.cmu.dynet.internal.Parameter;
import edu.cmu.dynet.internal.ParameterCollection;
import edu.cmu.dynet.internal.SimpleRNNBuilder;
import edu.cmu.dynet.internal.SimpleSGDTrainer;
import edu.cmu.dynet.internal.Tensor;

/**
 * 
 * @author Allan (allanmcgrady@gmail.com)
 * This follows the example in dynet RNN Tutorial: Character-level LSTM
 *  http://dynet.readthedocs.io/en/latest/tutorials_notebooks/RNNs.html
 *  
 *  Simply translating the code from Python to Java.
 */
public class LanguageModelExample {

	public static String characters = "abcdefghijklmnopqrstuvwxyz ";
	public static Random rand = new Random(1234);
	public int layers = 1;
	public int inputDim = 50;
	public int hiddenDim = 50;
	
	public LookupParameter ltp;
	public Parameter r;
	public Parameter bias;
	public Expression lt;
	public Expression re;
	public Expression be;
	
	public ComputationGraph cg;
	public SimpleRNNBuilder srnn;
	public Map<String, Integer> char2int;
	public Map<Integer, String> int2char;
	public SimpleSGDTrainer sgd;
	
	public LanguageModelExample(int layers, int inputDim, int hiddenDim) {
		//preprocessing to get the mapping between characters and index
		this.char2int = new HashMap<>();
		this.int2char = new HashMap<>();
		for (int i = 0 ; i < characters.length(); i++) {
			this.char2int.put(characters.substring(i, i+1), i);
			this.int2char.put(i, characters.substring(i, i+1));
		}
		this.char2int.put("<EOS>", characters.length());
		this.int2char.put(characters.length(), "<EOS>");
		this.layers = layers;
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.initializeModel();
	}
	
	static Dim makeDim(int[] dims) {
	    LongVector dimInts = new LongVector();
	    for (int i = 0; i < dims.length; i++) {
	      dimInts.add(dims[i]);
	    }
	    return new Dim(dimInts);
	}
	
	/**
	 * Initialize the model with parameters
	 * @return
	 */
	public ParameterCollection initializeModel() {
		DynetParams dp = new DynetParams();
		dp.setRandom_seed(1234);
		initialize(dp);
		ParameterCollection model = new ParameterCollection();
		sgd = new SimpleSGDTrainer(model);
		sgd.clip_gradients();
		sgd.setClip_threshold((float)5.0);
		cg = ComputationGraph.getNew();
		srnn = new SimpleRNNBuilder(layers, inputDim, hiddenDim, model);
		
		ltp =  model.add_lookup_parameters(characters.length() + 1, makeDim(new int[]{inputDim}));
		r = model.add_parameters(makeDim(new int[]{characters.length() + 1, hiddenDim}));
		bias = model.add_parameters(makeDim(new int[]{characters.length() + 1}));
		lt = parameter(cg, ltp);
		re = parameter(cg, r);
		be = parameter(cg, bias);
		return model;
	}
	
	/**
	 * Build the RNN for a specific sequence
	 * @param sent
	 * @return
	 */
	public Expression buildForward(String sent) {
		srnn.new_graph(cg);
		srnn.start_new_sequence();
		ExpressionVector finalErr = new ExpressionVector();
		String last = "<EOS>";
		String next = null;
		for (int i = 0 ; i <= sent.length(); i++) {
			Expression curr = lookup(cg, ltp, char2int.get(last));
			Expression curr_y = srnn.add_input(curr);
			Expression curr_r = exprPlus(exprTimes(re, curr_y), be);
			next = i == sent.length()? "<EOS>" :  sent.substring(i, i+1);
			Expression curr_err = pickneglogsoftmax(curr_r, char2int.get(next));
			finalErr.add(curr_err);
			last = next;
		}
		Expression lossExpr = sum_batches(sum((finalErr))); 
		return lossExpr;
	}
	
	public int sample(FloatVector fv) {
		float f = rand.nextFloat();
		int i = 0;
		for (i = 0; i < fv.size(); i++) {
			f -= fv.get(i);
			if (f <=0 ) break;
		}
		return i;
	}
	
	public String generateSentence() {
		srnn.new_graph(cg);
		srnn.start_new_sequence();
		Expression start = lookup(cg, ltp, char2int.get("<EOS>"));
		Expression s1 = srnn.add_input(start);
		String out = "";
		while(true) {
			Expression prob = softmax(exprPlus(exprTimes(re, s1), be) );
			int idx = sample(as_vector(cg.incremental_forward(prob)));
			out += int2char.get(idx);
			if (int2char.get(idx).equals("<EOS>")) break;
			s1 = srnn.add_input(lookup(cg, ltp, idx));
		}
		return out;
	}
	
	public static void main(String[] args) {
		int layers = 1;
		int inputDim = 50;
		int hiddenDim = 50;
		LanguageModelExample lm = new LanguageModelExample(layers, inputDim, hiddenDim);
		String sent = "a quick brown fox jumped over the lazy dog";
		float loss = 0;
		for (int it = 0; it < 100; it++) {
			Expression lossExpr = lm.buildForward(sent);
			Tensor lossTensor = lm.cg.forward(lossExpr);
			loss = as_scalar(lossTensor);
			lm.cg.backward(lossExpr);
			lm.sgd.update();
			if (it % 5 == 0) {
				System.out.print("loss is : " + loss);
				String prediction =  lm.generateSentence();
				System.out.println("   prediction: " + prediction);
			}
			lm.cg.forward(lossExpr);
			lm.cg.backward(lossExpr);
			lm.sgd.update();
		}
		
	}

}
