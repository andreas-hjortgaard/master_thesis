/* Conditional Random Fields for Object Localization
 * Master thesis source code
 * 
 * Authors:
 * Andreas Christian Eilschou (jwb226@alumni.ku.dk)
 * Andreas Hjortgaard Danielsen (gxn961@alumni.ku.dk)
 *
 * Department of Computer Science
 * University of Copenhagen
 * Denmark
 *
 * Date: 27-08-2012
 */

#ifndef _ESSWRAPPER_H_
#define _ESSWRAPPER_H_

#include "Types.h"

Bbox computeESS(const Image&, const Weights&);

#endif // _ESSWRAPPER_H_
