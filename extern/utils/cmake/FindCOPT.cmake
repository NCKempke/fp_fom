# FindCOPT.cmake -- this module finds the Cardinal optimizer
#
# Users might need to set/overwrite COPTDIR

# Find COPT headers and library
find_path(COPT_INCLUDE_DIR "copt.h" PATHS ${COPTDIR}/include/)
find_library(COPT_LIBRARY NAMES "libcopt.so" PATHS ${COPTDIR}/lib/)

mark_as_advanced(COPT_INCLUDE_DIR COPT_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(COPT DEFAULT_MSG COPT_LIBRARY COPT_INCLUDE_DIR)

if (COPT_FOUND)
	# Create imported target Copt::Copt
	add_library(Copt::Copt SHARED IMPORTED)
	set_target_properties(Copt::Copt PROPERTIES
		IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
		IMPORTED_LOCATION "${COPT_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${COPT_INCLUDE_DIR}"
		INTERFACE_LINK_LIBRARIES "${COPT_SHARED_LIBRARY}"
	)
endif()
