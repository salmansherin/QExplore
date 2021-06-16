let graphNodes=[];
let graphLinks =[];
let mappedLinks =[];


//mapping//line label//onclick listener



let style = cytoscape.stylesheet()
    .selector('node')
      .css({
		'shape':"rectangle",
        'height': 100,
        'width': 100,
        'background-fit': 'cover',
        'border-color': '#000',
        'border-width': 3,
        'border-opacity': 0.5,
		 "label": "data(label)",
		 'color':'red'
      })
    .selector('.eating')
      .css({
        'border-color': 'red'
      })
    .selector('.eater')
      .css({
        'border-width': 9
      })
    .selector('edge')
      .css({
		'line-cap' : 'butt',
        'curve-style': 'unbundled-bezier',
		'loop-sweep': '150',
        'width': 0.1,
        'target-arrow-shape': 'triangle',
        'line-color': '#000',
        'target-arrow-color': '#f00',
		"label": "data(linkName)",
		"font-size": "8px",
		"edge-text-rotation": "autorotate"
		
      });
for(let i=0;i<data.length;i++){
	if(data[i].src && data[i].src !== null && data[i].edges && data[i].edges !== null){
		
		style = style.selector('#' + data[i].src)
		  .css({
			'background-image': "./"+data[i].src+'.png',
		  });
			  
		  
		graphNodes.push({
			data:{id:data[i].src,label:""}
		});
		if(data[i].start == 1){
			graphNodes[graphNodes.length-1].data["label"] = "Root"
		}
		for(let j =0;j<data[i].edges.length;j++){
			//console.log('edge created')
			//if(data[i].src!=data[i].edges[j].state){
				graphLinks.push({
					data:{
						target:data[i].edges[j].state,
						source:data[i].src,
						label:data[i].edges[j].action,
						linkName: getActionName(data[i].edges[j].action),
					}
				});
			//}
		}
	}
}



for(let i=0;i<graphLinks.length;i++){
	let checkExists = false;
	for(let j=0;j<graphNodes.length;j++){
		if(graphLinks[i].data.target === graphNodes[j].data.id){
			checkExists = true;
			break;
		}
		if(!checkExists){
			graphNodes.push({data:{id:graphLinks[i].data.target}});
		}
	}
}

console.log(mappedLinks.length);

var cy = cytoscape({
  container: document.getElementById('cy'),

  boxSelectionEnabled: false,
  autounselectify: true,

  style: style,

  elements: {
    nodes: graphNodes,
    edges: graphLinks
  },

  layout: {
    //name: 'cose',
    directed: true,
    padding: 10
  }
});
// cy init



let selectedNode = null;
cy.on('tap', 'node', function(){
  if(selectedNode){
	selectedNode.removeClass("eater");  
  }
  var nodes = this;
  selectedNode = nodes;
  nodes.addClass('eater');
  let id = nodes[0].json().data.id;
  document.getElementById("myImg").src = id + ".png";
  document.getElementById("myNav").style.width = "100%";
  fetch(id + '.html')
  .then(response => response.text())
  .then(text => document.getElementById("data").innerHTML = safe_tags(text));
  
}); // on tap


function safe_tags(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;<br/>') ;
}

function getActionName(action){
	for(let i=0;i<mappedLinks.length;i++){
		if(action === mappedLinks[i].id){
			return mappedLinks[i].name
		}
	}
	mappedLinks.push({
		id:action,
		name:"Action " + (mappedLinks.length+1),
	});
    console.log(mappedLinks[mappedLinks.length-1].id)
	return mappedLinks[mappedLinks.length-1].name;
}
