use std::collections::HashSet;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, ToTokens};
use syn::{
    parse::Parse, parse_macro_input, parse_quote, punctuated::Punctuated, token::Comma, Attribute,
    Data, DeriveInput, Fields, GenericArgument, GenericParam, Generics, Ident, Lifetime,
    LifetimeParam, PathArguments, TraitBound, Type,
};

fn make_device_generics(generics: &Generics) -> (Generics, Vec<Type>) {
    let mut result = generics.clone();
    let mut changed_types = Vec::<Type>::new();
    result.params.iter_mut().for_each(|param| {
        if let GenericParam::Type(ref mut ty_param) = param {
            let ident = &ty_param.ident;
            for bound in &mut ty_param.bounds {
                if let syn::TypeParamBound::Trait(ref mut trait_bound) = bound {
                    let new_bound = (|| {
                        for path_segment in &mut trait_bound.path.segments {
                            if path_segment.ident == "gpu_builder" {
                                continue;
                            } else if path_segment.ident == "Builder" {
                                if let syn::PathArguments::AngleBracketed(ref mut args) =
                                    path_segment.arguments
                                {
                                    if args.args.len() != 1 {
                                        return None;
                                    } else if let syn::GenericArgument::Lifetime(_lifetime) =
                                        &args.args[0]
                                    {
                                        let param_type = parse_quote! { #ident };
                                        changed_types.push(param_type);
                                        return Some(TraitBound::from(
                                            syn::parse_quote! { gpu_builder::BuildResultType },
                                        ));
                                    } else {
                                        return None;
                                    }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                        None
                    })();
                    if let Some(new_bound) = new_bound {
                        *trait_bound = new_bound;
                    }
                }
            }
        }
    });
    result.params = result
        .params
        .into_iter()
        .filter(|param| {
            if let GenericParam::Lifetime(_lifetime) = param {
                false
            } else {
                true
            }
        })
        .collect();
    (result, changed_types)
}

fn make_host_generics(generics: &Generics, lifetime: &Lifetime) -> (Generics, Vec<Type>) {
    let mut result = generics.clone();
    let mut changed_types = Vec::<Type>::new();
    result.params.iter_mut().for_each(|param| {
        if let GenericParam::Type(ref mut ty_param) = param {
            let ident = &ty_param.ident;
            for bound in &mut ty_param.bounds {
                if let syn::TypeParamBound::Trait(ref mut trait_bound) = bound {
                    let new_bound = (|| {
                        for path_segment in &mut trait_bound.path.segments {
                            if path_segment.ident == "gpu_builder" {
                                continue;
                            } else if path_segment.ident == "BuildResultType" {
                                if let syn::PathArguments::None = path_segment.arguments {
                                    let param_type = parse_quote! { #ident };
                                    changed_types.push(param_type);
                                    return Some(TraitBound::from(
                                        syn::parse_quote! { gpu_builder::Builder<#lifetime> },
                                    ));
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                        None
                    })();
                    if let Some(new_bound) = new_bound {
                        *trait_bound = new_bound;
                    }
                }
            }
        }
    });
    (result, changed_types)
}

fn device_field_type(field_ty: &Type, changed_types: &[Type]) -> proc_macro2::TokenStream {
    let changed_types_tokens = changed_types
        .iter()
        .map(|ty| ty.to_token_stream().to_string())
        .collect::<Vec<_>>();
    if changed_types_tokens.contains(&field_ty.to_token_stream().to_string()) {
        quote! {
            #field_ty
        }
    } else {
        if let Type::Reference(type_reference) = field_ty {
            let elem = &type_reference.elem;
            quote! {
               <&'static #elem as gpu_builder::Builder<'static>>::Output
            }
        } else if let Type::Path(path) = field_ty {
            if path.path.segments.iter().any(|segment| {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    args.args.iter().any(|arg| {
                        if let GenericArgument::Lifetime(_lifetime) = arg {
                            true
                        } else {
                            false
                        }
                    })
                } else {
                    false
                }
            }) {
                //covert Ident to device version and remove all lifetime arguments
                let mut new_type = path.clone();
                if path.path.segments.len() != 1 {
                    panic!("Only single segment types with lifetimes are supported");
                }
                new_type.path.segments[0].ident = Ident::new(
                    &format!("{}Device", path.path.segments[0].ident),
                    Span::call_site(),
                );
                if let PathArguments::AngleBracketed(ref mut args) =
                    new_type.path.segments[0].arguments
                {
                    args.args = args
                        .args
                        .clone()
                        .into_iter()
                        .filter(|arg| {
                            if let GenericArgument::Lifetime(_lifetime) = arg {
                                false
                            } else {
                                true
                            }
                        })
                        .map(|mut arg| {
                            if let GenericArgument::Type(ref mut ty) = arg {
                                let device_type = device_field_type(ty, changed_types);
                                *ty = parse_quote! { #device_type };
                            }
                            arg
                        })
                        .collect();
                } else {
                    panic!("Expected angle bracketed arguments");
                };
                quote! {
                   #new_type
                }
            } else {
                quote! {
                    <#path as gpu_builder::Builder<'static>>::Output
                }
            }
        } else {
            quote! {
                <#field_ty as gpu_builder::Builder<'static>>::Output
            }
        }
    }
}

#[proc_macro_derive(DeviceStruct)]
pub fn derive_device_struct(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let attributes = &input.attrs;

    require_repr(&attributes);
    let attributes = filter_attributes(attributes);

    let visibility = &input.vis;
    let name = &input.ident;
    let device_name = Ident::new(&format!("{}Device", name), name.span());

    let generics = &input.generics;

    let (device_generics, changed_types) = make_device_generics(generics);

    // Process generics (preserve all parameters and where clauses)
    let (device_impl_generics, device_ty_generics, device_where_clause) =
        device_generics.split_for_impl();
    let device_from_host_ty_generics = device_generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(ty_param) => {
                let ident = &ty_param.ident;
                let param_type: Type = parse_quote! { #ident };
                let changed_types_tokens = changed_types
                    .iter()
                    .map(|ty| ty.to_token_stream().to_string())
                    .collect::<Vec<_>>();
                if changed_types_tokens.contains(&param_type.to_token_stream().to_string()) {
                    return quote! { #ident::Output };
                }
                quote! { #ident }
            }
            GenericParam::Lifetime(lifetime) => {
                quote! { #lifetime }
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                quote! { #ident }
            }
        })
        .collect::<Vec<_>>();
    let device_from_host_ty_generics = quote! { <#(#device_from_host_ty_generics),*> };
    let (host_impl_generics, host_ty_generics, host_where_clause) = generics.split_for_impl();

    let impl_block = match input.data {
        Data::Struct(data) => {
            // Handle structs (your existing code)
            let fields = match data.fields {
                Fields::Named(fields) => fields.named,
                _ => panic!("DeviceStruct derive for structs only works with named fields"),
            };
            let fields: Punctuated<_, Comma> = fields
                .into_iter()
                .filter(|field| {
                    if let Type::Path(path) = &field.ty {
                        for segment in path.path.segments.iter() {
                            if segment.ident == "core"
                                || segment.ident == "std"
                                || segment.ident == "marker"
                            {
                                continue;
                            } else if segment.ident == "PhantomData" {
                                return false;
                            } else {
                                return true;
                            }
                        }
                        true
                    } else {
                        true
                    }
                })
                .collect();

            let builder_fields = fields.iter().map(|field| {
                let field_name = &field.ident;
                let field_ty = &field.ty;
                let device_field_ty = device_field_type(field_ty, &changed_types);
                quote! {
                    #field_name: #device_field_ty
                }
            });
            let clone_fields = fields.iter().map(|field| {
                let field_name = &field.ident;
                quote! {
                    #field_name: self.#field_name.clone()
                }
            });

            quote! {
                #(#attributes)*
                #visibility struct #device_name #device_generics #device_where_clause {
                    #(#builder_fields),*
                }
                #[cfg(not(target_os = "cuda"))]
                unsafe impl #device_impl_generics cust::memory::DeviceCopy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Copy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Clone for #device_name #device_ty_generics #device_where_clause {
                    fn clone(&self) -> Self {
                        #device_name {
                            #(#clone_fields),*
                        }
                    }
                }
                impl #host_impl_generics From<&#device_name #device_from_host_ty_generics> for &#name #host_ty_generics #host_where_clause {
                    fn from(device: &#device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &*(device as *const #device_name #device_from_host_ty_generics as *const #name #host_ty_generics) }
                    }
                }
                impl #host_impl_generics From<&mut #device_name #device_from_host_ty_generics> for &mut #name #host_ty_generics #host_where_clause {
                    fn from(device: &mut #device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &mut *(device as *mut #device_name #device_from_host_ty_generics as *mut #name #host_ty_generics) }
                    }
                }

                #[cfg(target_os = "cuda")]
                impl #device_impl_generics gpu_builder::BuildResultType for #device_name #device_ty_generics #device_where_clause {}

                #[cfg(target_os = "cuda")]
                impl #host_impl_generics gpu_builder::Builder<'_> for #name #host_ty_generics #host_where_clause {
                    type Output = #device_name #device_from_host_ty_generics;
                }
            }
        }
        Data::Enum(data) => {
            let variants = data.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                let variant_index = &variant.discriminant.as_ref();
                let variant_index = variant_index.map(|(_, expr)| {
                    quote! { = #expr }
                });
                match &variant.fields {
                    Fields::Named(fields) => {
                        let fields = &fields.named;
                        let field_types = fields.iter().map(|f| {
                            let name = &f.ident;
                            let ty = &f.ty;
                            let device_ty = device_field_type(ty, &changed_types);
                            quote! { #name: #device_ty }
                        });
                        quote! {
                            #variant_name { #(#field_types),* } #variant_index
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let fields = &fields.unnamed;
                        let field_types = fields.iter().map(|f| {
                            let ty = &f.ty;
                            let device_ty = device_field_type(ty, &changed_types);
                            quote! { #device_ty }
                        });
                        quote! {
                            #variant_name ( #(#field_types),* ) #variant_index
                        }
                    }
                    Fields::Unit => quote! {
                        #variant_name #variant_index
                    },
                }
            });

            let variant_clone = data.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let field_names: Vec<_> = fields.named.iter().map(|f| &f.ident).collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                #name: #name.clone(),
                            }
                        });
                        quote! {
                            Self::#variant_name { #(#field_names),* } => {
                                #device_name::#variant_name {
                                    #(#fields_build)*
                                }
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = fields
                            .unnamed
                            .iter()
                            .enumerate()
                            .map(|(i, _f)| Ident::new(&format!("field_{}", i), name.span()))
                            .collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                #name.clone(),
                            }
                        });
                        quote! {
                            Self::#variant_name(#(#field_names),*) => {
                                #device_name::#variant_name(#(#fields_build)*)
                            },
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#variant_name => {
                                #device_name::#variant_name
                            },
                        }
                    }
                }
            });

            quote! {
                #(#attributes)*
                #visibility enum #device_name #device_generics #device_where_clause {
                    #(#variants),*
                }
                #[cfg(not(target_os = "cuda"))]
                unsafe impl #device_impl_generics cust::memory::DeviceCopy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Copy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Clone for #device_name #device_ty_generics #device_where_clause{
                    fn clone(&self) -> Self {
                        match self {
                            #(#variant_clone)*
                        }
                    }
                }
                impl #host_impl_generics From<&#device_name #device_from_host_ty_generics> for &#name #host_ty_generics #host_where_clause {
                    fn from(device: &#device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &*(device as *const #device_name #device_from_host_ty_generics as *const #name #host_ty_generics) }
                    }
                }
                impl #host_impl_generics From<&mut #device_name #device_from_host_ty_generics> for &mut #name #host_ty_generics #host_where_clause {
                    fn from(device: &mut #device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &mut *(device as *mut #device_name #device_from_host_ty_generics as *mut #name #host_ty_generics) }
                    }
                }

                #[cfg(target_os = "cuda")]
                impl #device_impl_generics gpu_builder::BuildResultType for #device_name #device_ty_generics #device_where_clause {}

                #[cfg(target_os = "cuda")]
                impl #host_impl_generics gpu_builder::Builder<'_> for #name #host_ty_generics #host_where_clause {
                    type Output = #device_name #device_from_host_ty_generics;
                }
            }
        }
        _ => panic!("DeviceStruct derive only works with structs and enums"),
    };

    TokenStream::from(impl_block)
}

#[proc_macro_derive(DeviceImpl, attributes(builder, use_lifetime, add_lifetime))]
pub fn derive_device_impl(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let impl_block = derive_device_impl_inner(input);
    impl_block.unwrap_or_else(|e| e.to_compile_error().into())
}

fn derive_device_impl_inner(input: DeriveInput) -> syn::Result<TokenStream> {
    let attributes = &input.attrs;

    require_repr(&attributes);

    let device_name = &input.ident;

    let mut name = Err(syn::Error::new_spanned(
            &input.ident,
            "Missing required `#[builder(SomeType)]` attribute. Please specify which type is its builder."
        ));

    let mut extra_lifetimes = get_extra_lifetimes("add_lifetime", &input.attrs)?;
    let builder_lifetime = get_extra_lifetimes("use_lifetime", &input.attrs)?;
    if builder_lifetime.len() != 1 {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "Expected exactly one lifetime attribute via `#[use_lifetime(\"'a\")]`",
        ));
    }
    let builder_lifetime = &builder_lifetime[0];

    for attr in &input.attrs {
        if attr.path().is_ident("builder") {
            if let syn::Meta::List(list) = &attr.meta {
                let path = syn::parse::Parser::parse2(syn::Path::parse, list.tokens.clone())?;

                if let Some(ident) = path.get_ident() {
                    name = Ok(ident.clone());
                } else {
                    return Err(syn::Error::new_spanned(
                        path,
                        "Expected an identifier as argument to `builder`",
                    ));
                }
            } else {
                return Err(syn::Error::new_spanned(
                    attr,
                    "Unexpected format. Expected an identifier like `builder(SomeType)`",
                ));
            }
        }
    }

    let name = name?;

    let generics = &input.generics;
    let mut host_generics = input.generics.clone();
    extra_lifetimes.reverse();
    host_generics.params.insert(
        0,
        GenericParam::Lifetime(syn::LifetimeParam::new(builder_lifetime.clone())),
    );
    for lifetime in extra_lifetimes {
        host_generics
            .params
            .insert(0, GenericParam::Lifetime(syn::LifetimeParam::new(lifetime)));
    }

    let (host_generics, changed_types) = make_host_generics(&host_generics, builder_lifetime);

    // Process generics (preserve all parameters and where clauses)
    let (device_impl_generics, device_ty_generics, device_where_clause) = generics.split_for_impl();
    let (host_impl_generics, host_ty_generics, host_where_clause) = host_generics.split_for_impl();

    let device_from_host_ty_generics = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(ty_param) => {
                let ident = &ty_param.ident;
                let param_type: Type = parse_quote! { #ident };
                let changed_types_tokens = changed_types
                    .iter()
                    .map(|ty| ty.to_token_stream().to_string())
                    .collect::<Vec<_>>();
                if changed_types_tokens.contains(&param_type.to_token_stream().to_string()) {
                    return quote! { #ident::Output };
                }
                quote! { #ident }
            }
            GenericParam::Lifetime(lifetime) => {
                quote! { #lifetime }
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                quote! { #ident }
            }
        })
        .collect::<Vec<_>>();
    let device_from_host_ty_generics = quote! { <#(#device_from_host_ty_generics),*> };

    let impl_block = match input.data {
        Data::Struct(data) => {
            // Handle structs (your existing code)
            let fields = match data.fields {
                Fields::Named(fields) => Ok(fields.named),
                _ => Err(syn::Error::new_spanned(
                    data.fields,
                    "DeviceImpl derive for structs only works with named fields",
                )),
            }?;

            let clone_fields = fields.iter().map(|field| {
                let field_name = &field.ident;
                quote! {
                    #field_name: self.#field_name.clone()
                }
            });

            quote! {
                #[cfg(not(target_os = "cuda"))]
                unsafe impl #device_impl_generics cust::memory::DeviceCopy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Copy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Clone for #device_name #device_ty_generics #device_where_clause {
                    fn clone(&self) -> Self {
                        #device_name {
                            #(#clone_fields),*
                        }
                    }
                }
                impl #host_impl_generics From<&#device_name #device_from_host_ty_generics> for &#name #host_ty_generics #host_where_clause {
                    fn from(device: &#device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &*(device as *const #device_name #device_from_host_ty_generics as *const #name #host_ty_generics) }
                    }
                }
                impl #host_impl_generics From<&mut #device_name #device_from_host_ty_generics> for &mut #name #host_ty_generics #host_where_clause {
                    fn from(device: &mut #device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &mut *(device as *mut #device_name #device_from_host_ty_generics as *mut #name #host_ty_generics) }
                    }
                }

                #[cfg(target_os = "cuda")]
                impl #device_impl_generics gpu_builder::BuildResultType for #device_name #device_ty_generics #device_where_clause {}

                #[cfg(target_os = "cuda")]
                impl #host_impl_generics gpu_builder::Builder<#builder_lifetime> for #name #host_ty_generics #host_where_clause {
                    type Output = #device_name #device_from_host_ty_generics;
                }
            }
        }
        Data::Enum(data) => {
            let variant_clone = data.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let field_names: Vec<_> = fields.named.iter().map(|f| &f.ident).collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                #name: #name.clone(),
                            }
                        });
                        quote! {
                            Self::#variant_name { #(#field_names),* } => {
                                #device_name::#variant_name {
                                    #(#fields_build)*
                                }
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = fields
                            .unnamed
                            .iter()
                            .enumerate()
                            .map(|(i, _f)| Ident::new(&format!("field_{}", i), name.span()))
                            .collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                #name.clone(),
                            }
                        });
                        quote! {
                            Self::#variant_name(#(#field_names),*) => {
                                #device_name::#variant_name(#(#fields_build)*)
                            },
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#variant_name => {
                                #device_name::#variant_name
                            },
                        }
                    }
                }
            });

            quote! {
                #[cfg(not(target_os = "cuda"))]
                unsafe impl #device_impl_generics cust::memory::DeviceCopy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Copy for #device_name #device_ty_generics #device_where_clause {}
                #[cfg(not(target_os = "cuda"))]
                impl #device_impl_generics Clone for #device_name #device_ty_generics #device_where_clause{
                    fn clone(&self) -> Self {
                        match self {
                            #(#variant_clone)*
                        }
                    }
                }
                impl #host_impl_generics From<&#device_name #device_from_host_ty_generics> for &#name #host_ty_generics #host_where_clause {
                    fn from(device: &#device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &*(device as *const #device_name #device_from_host_ty_generics as *const #name #host_ty_generics) }
                    }
                }
                impl #host_impl_generics From<&mut #device_name #device_from_host_ty_generics> for &mut #name #host_ty_generics #host_where_clause {
                    fn from(device: &mut #device_name #device_from_host_ty_generics) -> Self {
                        unsafe { &mut *(device as *mut #device_name #device_from_host_ty_generics as *mut #name #host_ty_generics) }
                    }
                }

                #[cfg(target_os = "cuda")]
                impl #device_impl_generics gpu_builder::BuildResultType for #device_name #device_ty_generics #device_where_clause {}

                #[cfg(target_os = "cuda")]
                impl #host_impl_generics gpu_builder::Builder<#builder_lifetime> for #name #host_ty_generics #host_where_clause {
                    type Output = #device_name #device_from_host_ty_generics;
                }
            }
        }
        _ => panic!("DeviceImpl derive only works with structs and enums"),
    };

    Ok(TokenStream::from(impl_block))
}

#[proc_macro_derive(Builder, attributes(use_lifetime))]
pub fn derive_builder(input_stream: TokenStream) -> TokenStream {
    let cloned_input_stream = input_stream.clone();
    let input = parse_macro_input!(cloned_input_stream as DeriveInput);
    let name = &input.ident;
    let device_name = Ident::new(&format!("{}Device", name), name.span());

    let generics = &input.generics;

    // Process generics (preserve all parameters and where clauses)
    let (_impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut builder_impl_generics = generics.clone();

    let extra_lifetimes = get_extra_lifetimes("use_lifetime", &input.attrs);
    if let Err(err) = extra_lifetimes {
        return err.to_compile_error().into();
    }
    let extra_lifetimes = extra_lifetimes.unwrap();
    let new_lifetime = if extra_lifetimes.len() == 1 {
        extra_lifetimes[0].clone()
    } else if extra_lifetimes.len() == 0 {
        let new_lifetime = generate_fresh_lifetime(&generics);
        builder_impl_generics.params.insert(
            0,
            GenericParam::Lifetime(LifetimeParam::new(new_lifetime.clone())),
        );
        new_lifetime
    } else {
        panic!("Only expected one lifetime parameter to use")
    };

    let (_device_generics, changed_types) = make_device_generics(generics);
    let device_ty_generics = generics
        .params
        .iter()
        .filter(|param| {
            if let GenericParam::Lifetime(_lifetime) = param {
                false
            } else {
                true
            }
        })
        .map(|param| match param {
            GenericParam::Lifetime(_lifetime_param) => {
                panic!("Unexpected lifetime parameter")
            }
            GenericParam::Type(ty_param) => {
                let ident = &ty_param.ident;
                let param_type: Type = parse_quote! { #ident };
                let changed_types_tokens = changed_types
                    .iter()
                    .map(|ty| ty.to_token_stream().to_string())
                    .collect::<Vec<_>>();
                if changed_types_tokens.contains(&param_type.to_token_stream().to_string()) {
                    return quote! { #ident::Output };
                } else {
                    return quote! { #ident };
                }
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                quote! { #ident }
            }
        })
        .collect::<Vec<_>>();
    let device_ty_generics = quote! { <#(#device_ty_generics),*> };

    let impl_block = match input.data {
        Data::Struct(data) => {
            // Handle structs (your existing code)
            let all_fields = match data.fields {
                Fields::Named(fields) => fields.named,
                _ => panic!("Builder derive for structs only works with named fields"),
            };
            let fields: Punctuated<_, Comma> = all_fields
                .clone()
                .into_iter()
                .filter(|field| {
                    if let Type::Path(path) = &field.ty {
                        for segment in path.path.segments.iter() {
                            if segment.ident == "core"
                                || segment.ident == "std"
                                || segment.ident == "marker"
                            {
                                continue;
                            } else if segment.ident == "PhantomData" {
                                return false;
                            } else {
                                return true;
                            }
                        }
                        true
                    } else {
                        true
                    }
                })
                .collect();
            let fields_tokens = fields
                .iter()
                .map(|field| field.to_token_stream().to_string())
                .collect::<Vec<_>>();
            let host_only_fields = all_fields
                .into_iter()
                .filter(|field| {
                    if fields_tokens.contains(&field.to_token_stream().to_string()) {
                        false
                    } else {
                        true
                    }
                })
                .collect::<Vec<_>>();

            let fields_device = &fields
                .iter()
                .map(|field| {
                    let field_name = field.ident.as_ref().unwrap();
                    syn::Ident::new(&format!("{}_device", field_name), field_name.span())
                })
                .collect::<Vec<_>>();

            let struct_fields = fields
                .iter()
                .map(|field| {
                    let field_name = &field.ident;
                    quote! {
                        #field_name: #field_name,
                    }
                })
                .collect::<Vec<_>>();

            let struct_fields_host = host_only_fields
                .iter()
                .map(|field| {
                    let field_name = &field.ident;
                    quote! {
                        #field_name: core::marker::PhantomData,
                    }
                })
                .collect::<Vec<_>>();

            let struct_fields_device = fields
                .iter()
                .zip(fields_device)
                .map(|(field, field_device)| {
                    let field_name = &field.ident;
                    quote! {
                        #field_name: #field_device,
                    }
                })
                .collect::<Vec<_>>();

            let build_fields = fields.iter().map(|field| {
                let field_name = &field.ident;
                quote! {
                    let #field_name = self.#field_name.build_inner(cache);
                }
            });

            let build_device_fields = fields.iter().zip(fields_device).map(|(field, field_device)| {
                let field_name = &field.ident;
                quote! {
                    let (#field_device, #field_name, buffers) = unsafe { self.#field_name.build_device_inner(stream, cache)? }.split();
                    device_buffer_list.combine(buffers);
                }
            });

            let copy_back_fields = fields.iter().map(|field| {
                let field_name = &field.ident;
                quote! {
                    let field_ptr = base_ptr + std::mem::offset_of!(#device_name #device_ty_generics, #field_name) as u64;
                    let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                    self.#field_name.copy_back(&field_device)?;
                }
            });

            let device_struct = derive_device_struct(input_stream);
            let device_struct = proc_macro2::TokenStream::from(device_struct);

            quote! {
                #device_struct

                #[cfg(not(target_os = "cuda"))]
                impl #builder_impl_generics gpu_builder::Builder<#new_lifetime> for #name #ty_generics #where_clause {
                    type Output = #device_name #device_ty_generics;
                    fn build_inner(self, cache: &mut gpu_builder::Cache<#new_lifetime>) -> <Self as gpu_builder::Builder<#new_lifetime>>::Output {
                        #(#build_fields)*
                        #device_name {
                            #(#struct_fields)*
                        }
                    }
                    unsafe fn build_device_inner(
                        self,
                        stream: &#new_lifetime cust::stream::Stream,
                        cache: &mut gpu_builder::Cache<#new_lifetime>,
                    ) -> cust::error::CudaResult<gpu_builder::BuildResult<#new_lifetime, Self>> {
                        let mut device_buffer_list = gpu_builder::DeviceBufferList::new();
                        #(#build_device_fields)*
                        let result_device = #device_name {
                            #(#struct_fields_device)*
                        };
                        let result_host = #name {
                            #(#struct_fields)*
                            #(#struct_fields_host)*
                        };
                        Ok(gpu_builder::BuildResult::new(result_device, result_host, stream, device_buffer_list))
                    }

                    fn copy_back(
                        &mut self,
                        c_device: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<#new_lifetime>>::Output>,
                    ) -> Result<(), cust::error::CudaError> {
                        let base_ptr = c_device.as_device_ptr().as_raw();

                        #(#copy_back_fields)*

                        Ok(())
                    }
                }
            }
        }
        Data::Enum(data) => {
            let device_enum = derive_device_struct(input_stream);
            let device_enum = proc_macro2::TokenStream::from(device_enum);

            let variant_build = data.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let field_names: Vec<_> = fields.named.iter().map(|f| &f.ident).collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                #name: #name.build_inner(cache),
                            }
                        });
                        quote! {
                            Self::#variant_name { #(#field_names),* } => {
                                #device_name::#variant_name {
                                    #(#fields_build)*
                                }
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = fields
                            .unnamed
                            .iter()
                            .enumerate()
                            .map(|(i, _f)| Ident::new(&format!("field_{}", i), name.span()))
                            .collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                #name.build_inner(cache),
                            }
                        });
                        quote! {
                            Self::#variant_name(#(#field_names),*) => {
                                #device_name::#variant_name(#(#fields_build)*)
                            },
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#variant_name => {
                                #device_name::#variant_name
                            },
                        }
                    }
                }
            });

            let variant_build_device = data.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let field_names: Vec<_> = fields.named.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                        let field_names_host: &Vec<_> = &field_names.iter().map(|field| Ident::new(&format!("{}_host", field), name.span())).collect();
                        let field_names_device: &Vec<_> = &field_names.iter().map(|field| Ident::new(&format!("{}_device", field), name.span())).collect();
                        let fields_build = field_names.iter().zip(field_names_host).zip(field_names_device).map(|((name, name_host), name_device)| {
                            quote! {
                                let (#name_device, #name_host, buffers) = #name.build_device_inner(stream, cache)?.split();
                                device_buffer_list.combine(buffers);
                            }
                        });
                        quote! {
                            Self::#variant_name { #(#field_names),* } => {
                                #(#fields_build)*
                                (#device_name::#variant_name {
                                    #(#field_names: #field_names_device),*
                                }, #name::#variant_name {
                                    #(#field_names: #field_names_host),*
                                })
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = fields
                            .unnamed
                            .iter()
                            .enumerate()
                            .map(|(i, _f)| Ident::new(&format!("field_{}", i), name.span()))
                            .collect();
                        let field_names_device: &Vec<_> = &field_names.iter().map(|name| Ident::new(&format!("{}_device", name), name.span())).collect();
                        let fields_build = field_names.iter().zip(field_names_device).map(|(name, name_device)| {
                            quote! {
                                let (#name_device, #name, buffers) = #name.build_device_inner(stream, cache)?.split();
                                device_buffer_list.combine(buffers);
                            }
                        });
                        quote! {
                            Self::#variant_name(#(#field_names),*) => {
                                #(#fields_build)*
                                (#device_name::#variant_name(#(#field_names_device),*), #name::#variant_name(#(#field_names),*))
                            },
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#variant_name => {
                                (#device_name::#variant_name, #name::#variant_name)
                            },
                        }
                    }
                }
            });

            let variant_copy_back = data.variants.iter().map(|variant| {
                let variant_name = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let field_names: Vec<_> = fields.named.iter().map(|f| &f.ident).collect();
                        let field_types: Vec<_> = fields.named.iter().map(|f| &f.ty).collect();
                        let fields_build = field_names.iter().map(|name| {
                            quote! {
                                let field_ptr = base_ptr + std::mem::offset_of!(#device_name #device_ty_generics, #variant_name.#name) as u64;
                                let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                                #name.copy_back(&field_device)?;
                            }
                        });
                        let fields_convert = field_names.iter().zip(field_types).map(|(name, ty)| {
                            quote! {
                                unsafe { std::ptr::read(Into::<&#ty>::into(&#name) as *const #ty) }
                            }
                        });
                        quote! {
                            #device_name::#variant_name { #(#field_names),* } => {
                                *self = #name::#variant_name{#(#field_names: #fields_convert),*};
                                match *self {
                                    #name::#variant_name{#(ref mut #field_names),*} => {
                                        #(#fields_build)*
                                    }
                                    _ => unreachable!(),
                                };
                            },
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let field_names: Vec<_> = fields
                            .unnamed
                            .iter()
                            .enumerate()
                            .map(|(i, _f)| Ident::new(&format!("field_{}", i), name.span()))
                            .collect();
                        let field_types: Vec<_> = fields.unnamed.iter().map(|f| &f.ty).collect();
                        let indexes: Vec<_> =
                            fields.unnamed.iter().enumerate().map(|(i, _f)| syn::Index::from(i)).collect();
                        let fields_build = field_names.iter().zip(indexes).map(|(name, index)| {
                            quote! {
                                let field_ptr = base_ptr + std::mem::offset_of!(#device_name #device_ty_generics, #variant_name.#index) as u64;
                                let field_device = unsafe { cust::memory::DeviceBox::from_raw(field_ptr) };
                                #name.copy_back(&field_device)?;
                            }
                        });
                        let fields_convert = field_names.iter().zip(field_types).map(|(name, ty)| {
                            quote! {
                                unsafe { std::ptr::read(Into::<&#ty>::into(&#name) as *const #ty) }
                            }
                        });
                        quote! {
                            #device_name::#variant_name(#(#field_names),*) => {
                                *self = #name::#variant_name(#(#fields_convert),*);
                                match *self {
                                    #name::#variant_name(#(ref mut #field_names),*) => {
                                        #(#fields_build)*
                                    }
                                    _ => unreachable!(),
                                };
                            },
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            #device_name::#variant_name => {
                                *self = #name::#variant_name;
                            },
                        }
                    }
                }
            });

            quote! {
                #device_enum

                #[cfg(not(target_os = "cuda"))]
                impl #builder_impl_generics gpu_builder::Builder<#new_lifetime> for #name #ty_generics #where_clause {
                    type Output = #device_name #device_ty_generics;
                    fn build_inner(self, cache: &mut gpu_builder::Cache<#new_lifetime>) -> <Self as gpu_builder::Builder<#new_lifetime>>::Output {
                        match self {
                            #(#variant_build)*
                        }
                    }
                    unsafe fn build_device_inner(
                        self,
                        stream: &#new_lifetime cust::stream::Stream,
                        cache: &mut gpu_builder::Cache<#new_lifetime>,
                    ) -> cust::error::CudaResult<gpu_builder::BuildResult<#new_lifetime, Self>> {
                        let mut device_buffer_list = gpu_builder::DeviceBufferList::new();
                        let (result_device, result_host) = match self {
                            #(#variant_build_device)*
                        };
                        Ok(gpu_builder::BuildResult::new(result_device, result_host, stream, device_buffer_list))
                    }

                    fn copy_back(
                        &mut self,
                        device_box: &cust::memory::DeviceBox<<Self as gpu_builder::Builder<#new_lifetime>>::Output>,
                    ) -> cust::error::CudaResult<()> {
                        let host_value = device_box.as_host_value()?;
                        let base_ptr = device_box.as_device_ptr().as_raw();
                        match host_value {
                            #(#variant_copy_back)*
                        }
                        Ok(())
                    }
                }
            }
        }
        _ => panic!("Builder derive only works with structs and enums"),
    };

    TokenStream::from(impl_block)
}

fn generate_fresh_lifetime(generics: &Generics) -> Lifetime {
    let existing: HashSet<String> = generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Lifetime(l) => Some(l.lifetime.ident.to_string()),
            _ => None,
        })
        .collect();
    let mut fresh = 1;
    while existing.contains(&format!("'a{}", fresh)) {
        fresh += 1;
    }
    Lifetime::new(&format!("'a{}", fresh), Span::call_site())
}

fn require_repr(attributes: &Vec<Attribute>) {
    let mut has_repr = false;
    for attr in attributes {
        if attr.path().is_ident("repr") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("C") || meta.path.is_ident("transparent") {
                    has_repr = true;
                }
                Ok(())
            });
        }
    }

    if !has_repr {
        panic!("DeviceStruct derive only works with structs and enums that are #[repr(C)] or #[repr(transparent)]");
    }
}

fn filter_attributes(attributes: &Vec<Attribute>) -> Vec<Attribute> {
    attributes
        .iter()
        .filter(|attr| {
            !(attr.path().is_ident("enum_dispatch")
                || attr.path().is_ident("builder")
                || attr.path().is_ident("use_lifetime"))
        })
        .cloned()
        .collect()
}

#[proc_macro_derive(DeviceCopyBuilder)]
pub fn derive_device_copy_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let attributes = &input.attrs;

    require_repr(&attributes);

    let name = &input.ident;

    let generics = &input.generics;
    // Process generics (preserve all parameters and where clauses)
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let impl_block = quote! {
        #[cfg(target_os = "cuda")]
        impl #impl_generics gpu_builder::Builder<'_> for #name #ty_generics #where_clause {
            type Output = #name #ty_generics;
        }
        #[cfg(target_os = "cuda")]
        impl #impl_generics gpu_builder::BuildResultType for #name #ty_generics #where_clause {}
    };

    TokenStream::from(impl_block)
}

fn get_extra_lifetimes(
    attr_name: &str,
    attrs: &Vec<Attribute>,
) -> Result<Vec<Lifetime>, syn::Error> {
    let mut extra_lifetimes = Vec::new();
    for attr in attrs {
        if attr.path().is_ident(attr_name) {
            if let syn::Meta::List(list) = &attr.meta {
                let tokens = list.tokens.clone().into_iter().collect::<Vec<_>>();

                if tokens.len() != 1 {
                    return Err(syn::Error::new_spanned(
                        list,
                        format!("Expected exactly one argument (Literal) to `{}`", attr_name),
                    ));
                }
                let token = &tokens[0];
                if let proc_macro2::TokenTree::Literal(lifetime) = token {
                    if let syn::Lit::Str(lifetime) = syn::Lit::new(lifetime.clone()) {
                        let lifetime = lifetime.value();

                        if lifetime.chars().next() != Some('\'') {
                            return Err(syn::Error::new_spanned(
                                token,
                                "Expected a lifetime starting with a tick (')",
                            ));
                        }

                        let lifetime = Lifetime::new(&lifetime, token.span());
                        extra_lifetimes.push(lifetime);
                    } else {
                        return Err(syn::Error::new_spanned(
                            token,
                            format!("Expected a literal as argument to `{}`", attr_name),
                        ));
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        token,
                        format!("Expected a literal as argument to `{}`", attr_name),
                    ));
                }
            } else {
                return Err(syn::Error::new_spanned(
                    attr,
                    format!(
                        "Unexpected format. Expected a lifetime like `{}(\"a\")`",
                        attr_name
                    ),
                ));
            }
        }
    }
    Ok(extra_lifetimes)
}
