Documentation
=============

Dynet uses Doxygen_ for commenting the code and Sphinx_ for the general documentation. 

If you're only documenting features you don't need to concern yourself with Sphinx, your doxygen comments will be integrated in the documentation automatically.

Doxygen guidelines
------------------

Please document any publicly accessible function you write using the doxygen syntax. 
You can see examples in the training_ file. The most important thing is to use :code:`/*` style comments and  :code:`\command` style commands.

For ease of access the documentation is divided into *groups*. For now the groups are optimizers and operations. If you implement a function that falls into one of these groups, add  :code:`\ingroup [group name]` at the beginning of your comment block.

If you want to create a group, use  :code:`\defgroup [group-name]` at the beginning of your file. Then create a file for this group in sphinx (see next section).

**Important** : You can use latex in doxygen comments with the syntax :code:`\f$ \f$`. For some reason since readthedocs updated their version of sphinx :code:`\f[ \f]` doesn't work anymore so *don't use it* it breaks the build.

Sphinx guidelines
-----------------

The sphinx source files are located in  :code:`doc/source`. They describe the documentation's organization using the reStructuredText_ Markup language.

Although reStructuredText is more powerful than Markdown_ it might feel less intuitive, especially when writing long documents. If needs be you can write your doc in Markdown and convert it using Pandoc_.

For a tutorial on Sphinx see their tutorial_.

Doxygen generated XML is integrated in sphinx files using the Breathe_ module. The only breathe command used now is :code:`doxygengroup`. You shouldn't used commands for individual classes/functions/structs without a good reason. Most information should be put in the doxygen comments.

Building the docs
-----------------

The documentation is automatically rebuilt by ReadTheDocs each time you push on Github.

If you want to build the documentation locally you'll need to install doxygen, sphinx and breathe and then run :code:`build_doc.sh` from the :code:`doc` folder.

.. _Doxygen: www.doxygen.org/
.. _Sphinx: http://www.sphinx-doc.org/en/stable/index.html
.. _training: https://github.com/clab/dynet/blob/master/dynet/training.h
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Markdown: https://daringfireball.net/projects/markdown/
.. _Pandoc: http://pandoc.org/
.. _tutorial: http://www.sphinx-doc.org/en/stable/tutorial.html
.. _Breathe: https://breathe.readthedocs.io/en/latest/
