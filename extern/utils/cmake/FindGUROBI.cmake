# FindGUROBI.cmake -- this module finds the GUROBI optimizer
#
# Users might need to set/overwrite GUROBIDIR

if(NOT DEFINED GUROBIDIR)
	if(DEFINED ENV{GUROBI_HOME})
		set(GUROBIDIR $ENV{GUROBI_HOME})
		message(STATUS "Setting GUROBIDIR from GUROBI_HOME environment variable: ${GUROBIDIR}")
	else()
		message(ERROR "GUROBIDIR is not set and GUROBI_HOME environment variable is not defined")
	endif()
else()
	message(STATUS "Using explicitly set GUROBIDIR: ${GUROBIDIR}")
endif()

# Optionally, you can check if the directory exists
if(NOT EXISTS ${GUROBIDIR})
	message(ERROR "GUROBI directory does not exist: ${GUROBIDIR}")
endif()

# Look for Gurobi 13!
# Find GUROBI headers and library
find_path(GUROBI_INCLUDE_DIR "gurobi_c++.h"
	PATHS
		${GUROBIDIR}/linux64cuda12/include/
		${GUROBIDIR}/armlinux64cuda12/include/
		${GUROBIDIR}/linux64/include/
		${GUROBIDIR}/armlinux64/include/
	NO_DEFAULT_PATH
)

# Find the C++ static library - look for the actual file name
find_library(GUROBI_CPP_LIBRARY
	NAMES
		"gurobi_g++8.5"  # Without "lib" prefix, CMake will add it
		"libgurobi_g++8.5.a"
		"gurobi_c++"     # Fallback to old name
		"libgurobi_c++.a"
	PATHS
		${GUROBIDIR}/linux64cuda12/lib/
		${GUROBIDIR}/armlinux64cuda12/lib/
		${GUROBIDIR}/linux64/lib/
		${GUROBIDIR}/armlinux64/lib/
	NO_DEFAULT_PATH
)

# Find the main shared library - use the versioned symlink
find_library(GUROBI_SHARED_LIBRARY
	NAMES
		"gurobi130"      # The symlink name
		"libgurobi130.so"
		"gurobi"         # Fallback
		"libgurobi.so"
	PATHS
		${GUROBIDIR}/linux64cuda12/lib/
		${GUROBIDIR}/armlinux64cuda12/lib/
		${GUROBIDIR}/linux64/lib/
		${GUROBIDIR}/armlinux64/lib/
	NO_DEFAULT_PATH
)


mark_as_advanced(GUROBI_INCLUDE_DIR GUROBI_CPP_LIBRARY GUROBI_SHARED_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_CPP_LIBRARY GUROBI_INCLUDE_DIR GUROBI_SHARED_LIBRARY)

if (GUROBI_FOUND)
	# Create imported target Gurobi::Gurobi
	add_library(Gurobi::Gurobi UNKNOWN IMPORTED)
	set_target_properties(Gurobi::Gurobi PROPERTIES
		IMPORTED_LOCATION "${GUROBI_CPP_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIR}"
		INTERFACE_LINK_LIBRARIES "${GUROBI_SHARED_LIBRARY}"
	)
endif()
