// semimarkov.h
// Author: Yangfeng Ji
// Date: 09-28-2016
// Time-stamp: <yangfeng 10/19/2017 13:41:41>

#ifndef ENTITYNLM_H
#define ENTITYNLM_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/tensor.h"
#include "dynet/cfsm-builder.h"
#include "dynet/pretrain.h"

#include "util.h"

// #include "../beam/beamsearch.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;


class EntityNLM {
  
private:
  LSTMBuilder builder;
  SoftmaxBuilder* smptr;
  // parameters for input
  LookupParameter p_X; // word embeddings
  
  // parameters for entity type
  Parameter p_W_R; // entity type prediction ({2, hidden_dim})
  
  // parameters for entity cluster
  Parameter p_W_E; // entity cluster prediction ({entity_dim, hidden_dim})
  Parameter p_lambda_dist;
  
  // parameters for entity length
  Parameter p_W_L; // length distribution parameter
  Parameter p_L_bias;

  // parameters for hidden -> entity transformation
  Parameter p_W_T;

  // parameter for entity embeddings
  Parameter p_E;

  // parameters for word prediction
  Parameter p_cont; // default context vector

  Parameter p_Wx;
  Parameter p_Te;
  Parameter p_Tc;

  // parameters for entity embedding update
  Parameter p_W_delta; // updating weight matrix
  unsigned ntype, menlen, indim, hidim, entdim;

  // word embeddings
  unordered_map<int, vector<float>> embeddings;
  bool with_embeddings;

  // ------------------------------------
  // graph-specific variables
  // expressions
  Expression WR, WL, L_bias, WE, lambda_dist, WT, embed_dummy, cont_dummy, Wdelta, Wx, Te, Tc, recip_norm;
  vector<Expression> entitylist;
  vector<float> entitydist;
  map<unsigned, unsigned> map_eidx_pos; // eidx to pos in entitylist mapping
  map<unsigned, unsigned> map_pos_eidx; // pos to eidx in entitylist mapping

  // ------------------------------------
  // string for sampled text
  string sampledtext;
  bool with_sample;

public:
  EntityNLM(){};
  EntityNLM(ParameterCollection& model,
	    unsigned vocab_size,
	    unsigned type_size,
	    unsigned men_length, // max mention length
	    Dict& d,
	    unsigned layers = 1,
	    unsigned input_dim = 32,
	    unsigned hidden_dim = 32,
	    unsigned entity_dim = 32,
	    string cluster_file="",
	    string fembed="");
  
  // init a CG
  int InitGraph(ComputationGraph& cg, float drop_rate);

  // generative model
  Expression BuildGraph(const Doc& doc,
			ComputationGraph& cg,
			Dict& d,
			int err_type=0,
			float err_weight=1.0,
			bool b_sample=false);

  // discriminative model
  Expression BuildDisGraph(const Doc& doc,
			   ComputationGraph& cg);

  // get sampled text
  string get_sampledtext();

private:
  int get_index(vector<float>& vec, bool take_zero=true);

  vector<float> get_dist_feat(vector<float> entitydist, int n);
  
  int create_entity(ComputationGraph&, Expression&,
		    vector<Expression>&, vector<float>&,
		    map<unsigned, unsigned>&,
		    map<unsigned, unsigned>&,
		    int, unsigned);
  
  int update_entity(ComputationGraph&, vector<Expression>&,
		    vector<float>&, map<unsigned, unsigned>&,
		    Expression&, Expression&, Expression&,
		    Expression&, int, unsigned);
};



#endif
